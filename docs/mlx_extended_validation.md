# Broader MLX GPU validation

This check tests Qwen3-4B scoring and decision behavior against the Hugging Face
backend on the same Mac. It uses handwritten inputs, not a task dataset. Matching
engines does not establish that their answers are correct.

Run it from the checkout with the existing MLX, Hugging Face, and development
extras installed. The runner also requires `psutil`. The installed distribution's
[license file](../bench/results_mlx/2026-09-28/psutil-7.2.2-LICENSE.txt)
identifies it as BSD-3-Clause. Install it with `python -m pip install psutil` if
it is absent from the validation environment.

```sh
.venv/bin/python scripts/mlx_extended_validation.py \
  --model /Users/pete/AnyJev/.local/native-model-4b \
  --json bench/results_mlx/2026-09-28/qwen3-4b-extended-float32.json
```

## Criteria set before the run

Both engines must use float32 on the GPU with identical checkpoint files and
scoring inputs. The runner disables PyTorch CPU fallback. A numerical pass needs
all scoring and decision argmaxes to agree, token log-probability differences at
most 0.1, and decision-probability differences at most 0.02. These are the original
parity tolerances. Every decision must have the requested level and a finite,
nonnegative probability distribution that sums to one.

The scoring fixtures contain exactly 1, 32, 128, 512, and 1024 tokens. They repeat
`x` to isolate sequence-length behavior. Requested token IDs include a duplicate.
The runner reverses both prompt order and requested token order, then compares
each output with the original call. The maximum permitted difference is 0.000001.

The stability phase warms every prompt length and mixed-call pattern, then runs
120 calls containing 200 prompt evaluations. It alternates single prompts,
different lengths, and duplicate prompts. The maximum permitted change from the
reference outputs is 0.000001. Across post-warmup samples, active GPU allocations
may grow by at most 64 MiB and process RSS by at most 512 MiB. The allocator uses
its default cache behavior. The runner records cached and peak GPU allocations
separately, then clears the cache after the stability phase to measure what it
retained. RSS alone does not describe unified-memory use.

The decision phase uses 24 new states and four question variants at raw, L0, and
L1. These cover choice, yes/no, score, and reversed choice options. The same eight
calibration examples from the original check remain separate from the new states.
Empty and whitespace-only states are included. Reversing options creates another
engine-comparison case; the check does not require invariance across option orders.

The runner stops if process RSS exceeds 20 GiB, sampled active GPU allocations
exceed 18 GiB, or total elapsed time exceeds 100 minutes. A resource stop leaves
partial evidence and cannot pass. Engines run sequentially. These ceilings limit
this experiment; they do not establish a supported model context length.

## Results

The [full result JSON](../bench/results_mlx/2026-09-28/qwen3-4b-extended-float32.json)
records a failed overall check. Both engines completed all planned cases, without
a resource stop. The numerical tolerances, call-independence checks, and sampled
memory limits passed. One near-tie score distribution failed the requirement that
every argmax match.

All measurements below come from that JSON. Total elapsed time was 316.50 seconds,
about 5 minutes 16 seconds, including model hashing and sequential worker startup.
MLX took 122.48 seconds and Hugging Face took 189.58 seconds. The repeated-call
phase occupied about 40 seconds after warmup. This was not a two-hour test.

| Check | Observed result | Criterion |
|---|---|---|
| Scoring lengths | All five completed, identical input tokens | All planned lengths |
| Maximum token log-probability difference | 0.0167084 | At most 0.1, passed |
| Maximum decision-probability difference | 0.00888075 | At most 0.02, passed |
| Decision top-bin agreement | 287 of 288 | All 288, failed |
| Valid raw/L0/L1 outputs | 288 per engine | All valid, passed |
| Reordered call difference | 0 for both engines | At most 0.000001, passed |
| Repeated MLX calls | 120 completed, 200 prompt evaluations, difference 0 | All complete and stable, passed |
| Maximum sampled active GPU allocation growth | 0 bytes | At most 64 MiB, passed |
| Maximum sampled RSS growth after warmup | 458,752 bytes | At most 512 MiB, passed |

MLX reported 16,089,872,396 bytes of active GPU allocations at every sampled point.
Its peak allocator measurement was 17,929,522,184 bytes. The stability phase ended
with 1,972,421,580 bytes cached; an explicit cache clear reduced cached allocations
to zero without changing active allocations. This distinguishes retained allocator
cache from growth in live tensors. The run saved 352 distinct decision-scoring
inputs, including exact prompts and token IDs, and they matched between engines.

### The near-tie case

For `Question.score("How urgent is this?", levels=["not", "low", "medium", "high"])`
at raw level, the state was `I paid yesterday but my account still says overdue.`
The exact distributions and public answers are in the full result JSON under
`backends.*.decisions`, where `question_index=2`, `state_index=0`, and `level="raw"`.

| Engine | Medium probability | High probability | Top-two margin | Returned numeric score |
|---|---|---|---|---|
| Hugging Face | 0.4976438721 | 0.5023434852 | 0.0046996130 | 2.5023308353 |
| MLX | 0.5037203532 | 0.4962666695 | 0.0074536837 | 2.4962536850 |

The public answer is a weighted numeric score. It changed by 0.0060771504; this is
not a different categorical public answer. The dominant bin changed across a
near tie, with maximum probability difference 0.0060768156. That is within the
numerical tolerance but still fails the predeclared all-argmax requirement.

The evidence does not establish a wrapper defect. It does establish that the
previous all-argmax result does not generalize to every new example. Keep this
limitation in any contribution description. The backend is functional on this
workload, but this broader validation is not an overall pass. No tolerance or
backend code changed to remove the failure.

## Limits

The repeated-call phase is a bounded workload, not a two-hour endurance test.
Memory samples cannot prove the absence of every leak. These checks do not cover
quantization, L2, concurrent clients, all supported architectures, or contexts
beyond 1024 tokens. Long prompts here are repetitive synthetic text. The broader
decision examples are short messages, not a long-document task benchmark.

The prior float16 log-probability tolerance failure remains in
[qwen3-4b-float16.json](../bench/results_mlx/2026-09-28/qwen3-4b-float16.json).
This experiment uses float32 and does not resolve that failure.

An initial attempt imposed a 512 MiB allocator cache limit. We stopped it because
that would not test default allocator behavior. Its partial evidence is preserved
in [the interrupted run](../bench/results_mlx/2026-09-28/qwen3-4b-extended-capped-partial.json)
and [worker output](../bench/results_mlx/2026-09-28/qwen3-4b-extended-capped-partial-mlx.json),
with the [source used for that attempt](../bench/results_mlx/2026-09-28/extended-capped-partial-source.py.txt).
It is incomplete and is not evidence of a pass.
