"""Compare MLX and HF on identical local weights, without holding both models in memory.

    python scripts/mlx_parity.py --model /path/to/model --dtype float32 --json parity.json

Check scoring and raw/L0/L1 outputs on handwritten fixtures. This does not measure
accuracy on a task dataset. Both engines use the same input tokens and all option
rotations, without sampling. Save log probabilities, decisions, environment details,
and file hashes in JSON. Use unquantized checkpoints to compare the same weights.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from anyjev import Decider, Question
from anyjev.readout import answer_labels, build_prompt, label_ids_for_perm, map_label_tokens, render_chat
from anyjev.state import render_state


def fixtures():
    questions = [Question.choice("Which team handles this?", ["billing", "technical", "sales", "other"]),
                 Question.noul("Is this message a complaint?"),
                 Question.score("How urgent is this?", levels=["not", "low", "medium", "high"])]
    calibration = ["My card was charged twice. Please fix this today.",
                   "The app keeps crashing. Nobody here can work.",
                   "Could you send pricing for a team of ten next week?",
                   "Thank you, everything is working now.",
                   "My refund has still not arrived after a month.",
                   "Production is down. We need technical assistance immediately.",
                   "I would like a demo sometime next month.",
                   "Just saying hello. No help needed."]
    labels = [[0, 1, 2, 3, 0, 1, 2, 3], [0, 0, 1, 1, 0, 0, 1, 1], [2, 3, 1, 0, 2, 3, 1, 0]]
    heldout = ["My card was charged twice.", "The app crashes on login.", "Do you offer bulk discounts?",
               "Thanks, all sorted now!", "URGENT: production is down for all users."]
    return questions, calibration, labels, heldout


def worker(args):
    started = time.monotonic()
    if args.worker == "hf":
        import torch

        from anyjev.backends.hf import HFBackend
        if not torch.backends.mps.is_available():
            raise RuntimeError("this validation requires the HF reference on MPS")
        be = HFBackend(args.model, device="mps", dtype=args.dtype, batch_size=1)
        device = str(next(be.model.parameters()).device)
        dtypes = sorted({str(p.dtype) for p in be.model.parameters()})
        if not device.startswith("mps"):
            raise RuntimeError(f"unexpected HF device: {device}")
    else:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from anyjev.backends.mlx import MLXBackend
        if not mx.metal.is_available():
            raise RuntimeError("this validation requires Metal")
        mx.set_default_device(mx.gpu)
        be = MLXBackend(args.model, dtype=args.dtype)
        device = str(mx.default_device())
        dtypes = sorted({str(p.dtype) for _, p in tree_flatten(be.model.parameters())})
    questions, calibration, labels, heldout = fixtures()
    prompts, ids = [], []
    for q in questions:
        tokens = map_label_tokens(be.tokenizer, answer_labels(q))
        perm = list(range(q.k))
        for state in heldout:
            prompts.append(render_chat(be.tokenizer, build_prompt(render_state(state), q, perm)))
            ids.append(label_ids_for_perm(q, tokens, perm))
    scored = be.next_token_logprobs(prompts, ids)
    levels, artifacts = {}, []
    for level in ("raw", "L0", "L1"):
        levels[level] = []
        for q, gold in zip(questions, labels):
            dec = Decider(be, shared_prefix=False, adaptive_shifts=False)
            if level == "L1":
                artifacts.append(dec.calibrate(q, calibration, gold))
            results = dec.decide_batch(heldout, q, level=level, require=level)
            for result in results:
                if result.level != level or not np.isfinite(result.probs).all():
                    raise AssertionError("invalid decision level or nonfinite probabilities")
                if np.any(result.probs < 0) or not np.isclose(result.probs.sum(), 1.0):
                    raise AssertionError("invalid probability distribution")
                levels[level].append({"level": result.level, "kind": q.kind,
                                      "probs": result.probs.tolist(), "answer": result.to_dict()["answer"]})
        print(f"{args.worker}: {level} checked", flush=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "transformers", "mlx", "mlx-lm")}
    result = {"backend": args.worker, "device": device, "parameter_dtypes": dtypes,
              "versions": versions, "elapsed_seconds": time.monotonic() - started,
              "prompts": prompts,
              "prompt_token_ids": [be.tokenizer.encode(p, add_special_tokens=False) for p in prompts],
              "label_token_ids": ids, "logprobs": [x.tolist() for x in scored], "levels": levels,
              "calibration_artifacts": artifacts}
    if args.worker == "mlx":
        result["peak_memory_bytes"] = mx.get_peak_memory()
    Path(args.json).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare(a, b, args):
    same_tokens = all(a[key] == b[key] for key in ("prompts", "prompt_token_ids", "label_token_ids"))
    differences = [float(np.max(np.abs(np.array(x) - y))) for x, y in zip(a["logprobs"], b["logprobs"])]
    scoring_agreement = sum(np.argmax(x) == np.argmax(y) for x, y in zip(a["logprobs"], b["logprobs"]))
    checks = {"identical_inputs": same_tokens, "max_abs_logprob_diff": max(differences),
              "scoring_argmax_agreement": int(scoring_agreement), "scoring_count": len(differences), "levels": {}}
    passed = same_tokens and max(differences) <= args.max_logprob_diff and scoring_agreement == len(differences)
    for level in ("raw", "L0", "L1"):
        x, y = a["levels"][level], b["levels"][level]
        delta = max(float(np.max(np.abs(np.array(p["probs"]) - q["probs"]))) for p, q in zip(x, y))
        agreement = sum(np.argmax(p["probs"]) == np.argmax(q["probs"]) for p, q in zip(x, y))
        checks["levels"][level] = {"max_abs_probability_diff": delta, "argmax_agreement": int(agreement),
                                   "count": len(x)}
        passed = passed and delta <= args.max_prob_diff and agreement == len(x)
    checks["passed"] = bool(passed)
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local, unquantized HF safetensors directory")
    ap.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
    ap.add_argument("--json", required=True)
    ap.add_argument("--max-logprob-diff", type=float, default=0.1)
    ap.add_argument("--max-prob-diff", type=float, default=0.02)
    ap.add_argument("--worker", choices=["hf", "mlx"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args)
        return
    model = Path(args.model).resolve()
    config = json.loads((model / "config.json").read_text())
    if config.get("quantization") or config.get("quantization_config"):
        ap.error("use unquantized weights for parity")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1")
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for backend in ("hf", "mlx"):
            output = Path(tmp) / f"{backend}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", backend,
                       "--model", str(model), "--dtype", args.dtype, "--json", str(output)]
            subprocess.run(command, check=True, env=env, cwd=root)
            results[backend] = json.loads(output.read_text())
    hashes = {p.name: digest(p) for p in sorted(model.iterdir())
              if p.suffix in (".json", ".safetensors")}
    source_hashes = {str(p.relative_to(root)): digest(p) for p in
                     (root / "anyjev/backends/mlx.py", root / "scripts/mlx_parity.py")}
    report = {"purpose": "handwritten scoring and decision-contract parity; not a task-accuracy benchmark",
              "command": sys.argv, "model_path": str(model), "model_file_sha256": hashes,
              "source_sha256": source_hashes, "dtype": args.dtype, "batch_size": 1,
              "python": sys.version, "platform": platform.platform(),
              "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root).strip(),
              "max_logprob_diff_tolerance": args.max_logprob_diff,
              "max_probability_diff_tolerance": args.max_prob_diff,
              "fixtures": {"calibration": fixtures()[1], "labels": fixtures()[2], "heldout": fixtures()[3]},
              "backends": results, "checks": compare(results["hf"], results["mlx"], args)}
    dest = Path(args.json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["checks"], indent=2))
    if not report["checks"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
