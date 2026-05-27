"""Benchmark TQFT topological protection (Phase II.2)."""
import json
import os
import time

import numpy as np
import torch

from revo.tqft import TQFTConfig, TopologicalProtector


def main():
    V = 32000
    N = 200
    TOPK = 64
    NUM_BRAIDS = 5

    cfg = TQFTConfig(topk=TOPK, num_braids=NUM_BRAIDS, noise_sigma=0.05, seed=42)
    protector = TopologicalProtector(cfg)

    rng = np.random.RandomState(0)
    logits_list = [torch.from_numpy(rng.randn(V).astype(np.float32)) for _ in range(N)]
    target_ids = [int(rng.randint(0, V)) for _ in range(N)]

    nll_unprotected = []
    nll_protected = []
    variances = []

    t0 = time.perf_counter()
    for logits, tid in zip(logits_list, target_ids):
        logp = torch.log_softmax(logits, dim=-1)
        nll_unprotected.append(-logp[tid].item())

        protected_lp = protector.protect_logprobs(logits)
        nll_protected.append(-protected_lp[tid].item())

        variances.append(protector.logical_error(logits))
    elapsed = time.perf_counter() - t0

    result = {
        "benchmark": "TQFT topological protection",
        "vocab_size": V,
        "n_samples": N,
        "config": {"topk": TOPK, "num_braids": NUM_BRAIDS, "noise_sigma": 0.05},
        "nll_unprotected": float(np.mean(nll_unprotected)),
        "nll_protected": float(np.mean(nll_protected)),
        "nll_diff": float(np.mean(nll_protected)) - float(np.mean(nll_unprotected)),
        "variance_mean": float(np.mean(variances)),
        "variance_std": float(np.std(variances)),
        "variance_min": float(np.min(variances)),
        "variance_max": float(np.max(variances)),
        "avg_time_us": elapsed / N * 1e6,
    }

    out_path = os.path.join("quality", "phase2_runs", "ii2_tqft.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
