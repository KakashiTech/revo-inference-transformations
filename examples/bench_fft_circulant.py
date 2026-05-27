from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn

from revo.fft_kernel import CirculantLinear, nearest_circulant_first_column


def benchmark_size(N: int, iters: int = 100):
    torch.manual_seed(42)
    W = torch.randn(N, N)
    x = torch.randn(1, N)

    # nearest_circulant_first_column
    t0 = time.perf_counter()
    for _ in range(iters):
        c = nearest_circulant_first_column(W)
    ncfc_time = (time.perf_counter() - t0) / iters * 1000

    # CirculantLinear forward
    cl = CirculantLinear(N, bias=True)
    cl.set_from_weight(W)
    cl.eval()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = cl(x)
    circ_time = (time.perf_counter() - t0) / iters * 1000

    # Plain Linear forward
    lin = nn.Linear(N, N, bias=True)
    with torch.no_grad():
        lin.weight.copy_(W)
    lin.eval()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = lin(x)
    linear_time = (time.perf_counter() - t0) / iters * 1000

    return {
        "size": N,
        "ncfc_time_ms": round(ncfc_time, 4),
        "circulant_linear_time_ms": round(circ_time, 4),
        "linear_time_ms": round(linear_time, 4),
        "circulant_speedup_vs_linear": round(linear_time / circ_time, 4) if circ_time > 0 else None,
        "ncfc_speedup_vs_linear": round(linear_time / ncfc_time, 4) if ncfc_time > 0 else None,
    }


def main():
    sizes = [64, 128, 256]
    reports = []
    for N in sizes:
        r = benchmark_size(N, iters=100)
        reports.append(r)
        print(f"  N={N:3d}  ncfc={r['ncfc_time_ms']:8.4f}ms  "
              f"circ={r['circulant_linear_time_ms']:8.4f}ms  "
              f"linear={r['linear_time_ms']:8.4f}ms  "
              f"circ_speedup={r['circulant_speedup_vs_linear']:.2f}x")

    results = {
        "benchmark": "FFT Circulant vs Linear",
        "description": (
            "Compares nearest_circulant_first_column (NCFC), "
            "CirculantLinear forward, and plain Linear forward."
        ),
        "by_size": reports,
        "reference_phase3": {
            "note": "Phase 3 used circulant replacement on gpt2 layers; see quality/phase3_runs/*.json",
            "sample_file": "quality/phase3_runs/phase3_1769130337.json",
            "sample_metrics": {
                "baseline_nll": 5.5872,
                "with_phase3_nll": 5.5846,
                "eval_speedup": 2.35,
            },
        },
    }

    out_dir = os.path.join("quality", "phase1_runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "i5_fft_circulant.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
