# MLX backend

From this checkout on Apple silicon, run `pip install -e '.[mlx]'`. The backend
loads local safetensors or a model supported by `mlx-lm` from Hugging Face.

```python
from anyjev import Decider, Question
from anyjev.backends.mlx import MLXBackend

backend = MLXBackend("Qwen/Qwen3-4B", dtype="float16")
question = Question.noul("Is this message a complaint?")
decider = Decider(backend)
print(decider.decide("My delivery is two weeks late.", [question], level="L0"))
```

`dtype=None` keeps the checkpoint's floating dtype. Use `revision=` to pin a Hub
revision. The loader disables remote tokenizer code by default.

The backend computes log-softmax over the full vocabulary and returns the requested
tokens' log probabilities. It never samples or generates tokens. For L1, first call
`decider.calibrate(question, states, labels)` with labelled examples.

Prompts run one at a time without padding. Shared-prefix caching and L2 are not
implemented. Long prompts use more memory because the model projects every position
to vocabulary logits. The checks below cover unquantized weights. Quantized models
need separate calibration and validation.

## GPU validation

Install the test dependencies with `pip install -e '.[mlx,hf,dev]'`. Use the same
local checkpoint for both engines. The parity command runs Hugging Face on MPS and
MLX on its GPU in separate processes. It disables PyTorch CPU fallback and records
actual devices, parameter dtypes, input tokens, dependency versions, and file hashes.

```sh
python scripts/mlx_parity.py --model /path/to/model --dtype float32 --json parity.json
ANYJEV_MLX_MODEL=/path/to/model pytest -q tests/test_mlx.py -m engine
```

The script compares token log probabilities and raw/L0/L1 decision distributions
on handwritten examples. Both engines fit L1 on the same calibration examples,
which are separate from the evaluation examples. This checks backend agreement
and valid outputs; it does not measure task accuracy. The GPU smoke test runs only
when `ANYJEV_MLX_MODEL` names a model and Metal is available.

## Recorded Mac checks

These runs used an M5 Pro with 24 GB unified memory. Hugging Face ran on `mps:0`
and MLX on `Device(gpu, 0)`. Both engines used the same checkpoint files, rendered
prompts, and token ids. A pass requires matching argmaxes, a token log-probability
difference no greater than 0.1, and a decision-probability difference no greater
than 0.02. These tolerances apply to this check.

| Model / dtype | Max log-probability difference | Max decision-probability difference | Decision argmax matches | Verdict and source |
|---|---|---|---|---|
| Qwen3-0.6B / float32 | 0.05037 | 0.00553 | 45/45 | [Pass](../bench/results_mlx/2026-09-28/qwen3-0.6b-float32.json) |
| Qwen3-4B / float16 | 0.28125 | 0.01512 | 45/45 | [Fails log-probability tolerance](../bench/results_mlx/2026-09-28/qwen3-4b-float16.json) |
| Qwen3-4B / float32 | 0.07608 | 0.00736 | 45/45 | [Pass](../bench/results_mlx/2026-09-28/qwen3-4b-float32.json) |
| SmolLM2-135M-Instruct / float32 | 0.00891 | 0.00645 | 45/45 | [Pass](../bench/results_mlx/2026-09-28/smollm2-135m-float32.json) |

Each source contains 15 scoring prompts and 15 decisions at each of raw, L0, and L1.
The checkpoint revisions are `c1899de289a04d12100db370d81485cdf75e47ca` for 0.6B and
`1cfa9a7208912126459214e8b04321603b3df60c` for 4B; each JSON also hashes the local weights.
All runs used MLX 0.32.2, mlx-lm 0.29.1, torch 2.14.0, and transformers 4.57.3,
as recorded in those JSONs. SmolLM2 exercises the Llama implementation and a different
tokenizer from Qwen3. Its [source record](../bench/results_mlx/2026-09-28/smollm2-135m-source.json)
pins revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` and records the Apache-2.0 license.
SmolLM2 passed with the same fixtures and tolerances, without backend changes. Its separate
[GPU smoke result](../bench/results_mlx/2026-09-28/smollm2-135m-smoke.json) checks direct
scoring and raw/L0/L1 outputs. We have not tested other models or quantized weights.

A [single-prompt diagnostic](../bench/results_mlx/2026-09-28/qwen3-4b-precision-probe.json)
reproduces the float16 difference through direct model calls. The MLX wrapper returns
exactly the same values as its direct call. Float32 reduces the difference between
engines, which points to numerical precision. Use float32 for this parity check.
Float16 returned valid decisions on these examples but failed the token
log-probability tolerance.

The [validation log](../bench/results_mlx/2026-09-28/validation.json) records a clean
ruff check and 122 passing tests with the Qwen3-4B GPU smoke enabled. A separate
numpy-only environment passed 114 tests and skipped eight optional-engine tests.
The existing truncation test also needed `accelerate` in the full environment.
The MLX backend does not depend on it.

The [prose-edit record](../bench/results_mlx/2026-09-28/prose-edit.json) preserves the
source used for these runs. Later docstring edits changed the file hashes but left
the executable code unchanged. The original GPU result files remain intact.
