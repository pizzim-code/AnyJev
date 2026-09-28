"""Check MLX scoring with numpy stubs and an optional GPU test with real weights."""
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runtime(monkeypatch):
    mx = ModuleType("mlx.core")
    mx.array, mx.float32, mx.float16, mx.bfloat16 = np.array, np.float32, np.float16, "bfloat16"
    mx.logsumexp = lambda x: np.logaddexp.reduce(x)
    mx.eval = lambda *args: None
    package = ModuleType("mlx")
    package.core = mx
    calls = []
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [ord(c) % 4 for c in text])

    class Model:
        def __init__(self):
            self.dtype = None
            self.evaluating = False

        def eval(self):
            self.evaluating = True

        def set_dtype(self, dtype):
            self.dtype = dtype

        def parameters(self):
            return []

        def __call__(self, inputs):
            calls.append(inputs.tolist())
            # The final token changes the answer. Earlier positions are deliberately different.
            rows = [[1000.0, 1001.0, 999.0, 1010.0 + token] for token in inputs[0]]
            return np.array([rows], dtype=np.float32)

    model = Model()
    seen = {}
    lm = ModuleType("mlx_lm")

    def load(name, **kwargs):
        seen.update(name=name, **kwargs)
        return model, tokenizer

    lm.load = load
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", lm)
    return SimpleNamespace(model=model, calls=calls, seen=seen, tokenizer=tokenizer)


def test_full_vocabulary_normalization_order_and_duplicates(runtime):
    from anyjev.backends.mlx import MLXBackend

    be = MLXBackend("local/model", revision="pinned", dtype="float32")
    out = be.next_token_logprobs(["ab", "c"], [[2, 0, 2], [3, 1]])
    for values, last, ids in zip(out, [2, 3], [[2, 0, 2], [3, 1]]):
        logits = np.array([1000, 1001, 999, 1010 + last], dtype=float)
        expected = (logits - np.logaddexp.reduce(logits))[ids]
        np.testing.assert_allclose(values, expected, atol=1e-4)
        assert values.dtype == np.float64
    assert np.exp(out[0]).sum() < 0.01  # Must not normalize over requested labels only.
    assert runtime.calls == [[[1, 2]], [[3]]]
    assert runtime.model.evaluating
    assert runtime.model.dtype == np.float32
    assert runtime.seen["revision"] == "pinned"
    assert runtime.seen["tokenizer_config"]["trust_remote_code"] is False
    assert be.tokenizer is runtime.tokenizer
    assert be.name == "local/model"


def test_prompts_are_independent(runtime):
    from anyjev.backends.mlx import MLXBackend

    be = MLXBackend("local/model")
    first = be.next_token_logprobs(["a", "bbb", "a"], [[0, 3]] * 3)
    np.testing.assert_array_equal(first[0], first[2])
    np.testing.assert_array_equal(first[0], be.next_token_logprobs(["a"], [[0, 3]])[0])


def test_empty_batch_and_label_set(runtime):
    from anyjev.backends.mlx import MLXBackend

    be = MLXBackend("local/model")
    assert be.next_token_logprobs([], []) == []
    assert be.next_token_logprobs(["a"], [[]])[0].shape == (0,)


@pytest.mark.parametrize("prompts,ids", [(["a"], []), ([""], [[0]]),
                                         (["a"], [[-1]]), (["a"], [[4]]),
                                         (["a"], [[1.5]]), (["a"], [[True]])])
def test_invalid_inputs_fail(runtime, prompts, ids):
    from anyjev.backends.mlx import MLXBackend

    with pytest.raises(ValueError):
        MLXBackend("local/model").next_token_logprobs(prompts, ids)


def test_invalid_dtype_fails_before_loading(runtime):
    from anyjev.backends.mlx import MLXBackend

    with pytest.raises(ValueError, match="dtype"):
        MLXBackend("local/model", dtype="int8")
    assert not runtime.seen


@pytest.mark.engine
def test_real_mlx_scoring_and_levels():
    path = os.environ.get("ANYJEV_MLX_MODEL")
    if not path:
        pytest.skip("set ANYJEV_MLX_MODEL to a local model path")
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("the engine smoke requires a Metal GPU")
    mx.set_default_device(mx.gpu)
    from anyjev import Decider, Question
    from anyjev.backends.mlx import MLXBackend

    be = MLXBackend(path, dtype="float16")
    prompt = "The light is on."
    tokens = be.tokenizer.encode(prompt, add_special_tokens=False)
    logits = be.model(mx.array([tokens]))[0, -1, :].astype(mx.float32)
    full = np.array(logits - mx.logsumexp(logits))
    ids = [int(np.argmax(full)), 0, len(full) - 1, 0]
    scores = be.next_token_logprobs([prompt, "A different prompt.", prompt], [ids] * 3)
    np.testing.assert_allclose(scores[0], full[ids], atol=1e-5)
    np.testing.assert_allclose(scores[0], scores[2], atol=1e-5)
    q = Question.noul("Is the light on?")
    states = ["The light is on.", "The light is off."] * 4
    dec = Decider(be, adaptive_shifts=False)
    for level in ("raw", "L0", "L1"):
        if level == "L1":
            dec.calibrate(q, states, [0, 1] * 4)
        results = dec.decide_batch(["The lamp is lit.", "The lamp is unlit."], q, level=level)
        for result in results:
            assert result.level == level
            assert np.isfinite(result.probs).all()
            np.testing.assert_allclose(result.probs.sum(), 1.0)
