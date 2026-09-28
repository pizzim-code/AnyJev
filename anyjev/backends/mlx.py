"""Native MLX scoring for raw / L0 / L1, with no token generation.

From the checkout, install ``pip install -e '.[mlx]'`` on Apple silicon.
Use a model supported by mlx-lm, stored as local HF safetensors or an MLX checkpoint.
For a comparison against HFBackend, use the same unquantized weights and dtype.
"""
from __future__ import annotations

from numbers import Integral
from typing import List, Sequence

import numpy as np


class MLXBackend:
    def __init__(self, model_name: str, *, dtype: str | None = None,
                 revision: str | None = None, trust_remote_code: bool = False):
        """Load a model and tokenizer. None preserves the checkpoint's floating dtype.

        Each prompt runs separately, without padding or a shared cache. The model returns
        logits directly. L2 and shared-prefix scoring are not implemented.
        Long prompts use more memory because the model projects every position to logits.
        """
        if dtype not in (None, "float16", "bfloat16", "float32"):
            raise ValueError("dtype must be None, float16, bfloat16, or float32")
        import mlx.core as mx
        from mlx_lm import load

        self.name, self.revision, self.dtype = model_name, revision, dtype
        self.model, self.tokenizer = load(
            model_name, revision=revision,
            tokenizer_config={"trust_remote_code": trust_remote_code})
        self.model.eval()
        if dtype is not None:
            self.model.set_dtype(getattr(mx, dtype))
            mx.eval(self.model.parameters())

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        """Return requested ids in order from the last position's full-vocabulary log-softmax.

        Compute log-softmax in float32 and return float64 arrays, as HFBackend does.
        Raise ValueError for an empty prompt or an invalid token id.
        """
        import mlx.core as mx

        if len(prompts) != len(token_ids):
            raise ValueError("prompts and token_ids must have the same length")
        results = []
        for prompt, ids in zip(prompts, token_ids):
            ids = list(ids)
            if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in ids):
                raise ValueError("token ids must be nonnegative integers")
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            if not tokens:
                raise ValueError("a prompt must encode to at least one token")
            logits = self.model(mx.array([tokens]))[0, -1, :].astype(mx.float32)
            if any(i >= logits.shape[-1] for i in ids):
                raise ValueError("token id is outside the model vocabulary")
            if not ids:
                results.append(np.empty(0, dtype=np.float64))
                continue
            logprobs = logits - mx.logsumexp(logits)
            selected = logprobs[mx.array(ids)]
            mx.eval(selected)
            results.append(np.array(selected, dtype=np.float64))
        return results
