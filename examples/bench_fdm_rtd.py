from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn

from revo.fdm_rtd import FDMFilterBank


def main():
    n_bands = 4
    in_features = 128
    out_features = 128
    batch_size = 2

    torch.manual_seed(42)
    fdm = FDMFilterBank(n_bands, in_features, out_features)
    fdm.eval()

    linear = nn.Linear(in_features, out_features, bias=True)
    linear.eval()

    x = torch.randn(batch_size, in_features)

    warmup = 100
    timed = 100

    # Warmup
    for _ in range(warmup):
        _ = fdm(x)
        _ = linear(x)

    # Timed FDM
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(timed):
        _ = fdm(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fdm_time = time.perf_counter() - t0

    # Timed Linear
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(timed):
        _ = linear(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    linear_time = time.perf_counter() - t0

    fdm_time_ms = fdm_time / timed * 1000
    linear_time_ms = linear_time / timed * 1000
    throughput_ratio = linear_time / fdm_time if fdm_time > 0 else float("inf")

    results = {
        "config": {
            "input_shape": [batch_size, in_features],
            "output_shape": [batch_size, out_features],
            "n_bands": n_bands,
            "warmup_iters": warmup,
            "timed_iters": timed,
        },
        "fdm_time_ms": round(fdm_time_ms, 4),
        "linear_time_ms": round(linear_time_ms, 4),
        "throughput_ratio": round(throughput_ratio, 4),
        "target_grh": "throughput >= 1.5x",
        "target_met": throughput_ratio >= 1.5,
    }

    out_dir = os.path.join("quality", "phase1_runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "i4_fdm_rtd.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
