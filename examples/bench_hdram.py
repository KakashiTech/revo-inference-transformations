"""Benchmark HDRAM recall@1 for exact and noisy queries (Phase I.3)."""
import json
import os
import time

import numpy as np

from revo.archive.hdram import HDRAM, HDRAMConfig


def main():
    N = 1000
    KEY_DIM = 64
    VAL_DIM = 64
    MAX_ENTRIES = 2000
    SIGMAS = [0.01, 0.05, 0.1]

    rng = np.random.RandomState(42)
    keys = [rng.randn(KEY_DIM).astype(np.float32) for _ in range(N)]
    vals = [rng.randn(VAL_DIM).astype(np.float32) for _ in range(N)]

    cfg = HDRAMConfig(key_dim=KEY_DIM, val_dim=VAL_DIM, max_entries=MAX_ENTRIES)
    hdram = HDRAM(cfg)

    t0 = time.perf_counter()
    for k, v in zip(keys, vals):
        hdram.put(k, v)
    put_time = time.perf_counter() - t0
    print(f"Stored {hdram.size} entries in {put_time * 1e3:.1f} ms")

    # exact query recall@1
    exact_hits = 0
    query_times = []
    for k, expected_v in zip(keys, vals):
        t0 = time.perf_counter()
        results = hdram.query(k, topk=1)
        elapsed = time.perf_counter() - t0
        query_times.append(elapsed)
        if results:
            retrieved, score = results[0]
            if score >= cfg.recall_threshold:
                exact_hits += 1
    exact_recall = exact_hits / N if N > 0 else 0.0
    avg_query_time_us = float(np.mean(query_times) * 1e6)
    print(f"Exact recall@1: {exact_recall:.4f}  |  avg query {avg_query_time_us:.2f} us")

    # noisy query recall@1 at various sigmas
    noisy_recalls = {}
    for sigma in SIGMAS:
        hits = 0
        for k, expected_v in zip(keys, vals):
            noisy_k = k + sigma * rng.randn(KEY_DIM).astype(np.float32)
            results = hdram.query(noisy_k, topk=1)
            if results:
                retrieved, score = results[0]
                if score >= cfg.recall_threshold:
                    hits += 1
        recall = hits / N if N > 0 else 0.0
        noisy_recalls[str(sigma)] = recall
        print(f"  sigma={sigma:.2f} recall@1 = {recall:.4f}")

    result = {
        "benchmark": "HDRAM recall@1",
        "n_keys": N,
        "key_dim": KEY_DIM,
        "val_dim": VAL_DIM,
        "max_entries": MAX_ENTRIES,
        "exact_recall": exact_recall,
        "noisy_recall": noisy_recalls,
        "avg_query_time_us": avg_query_time_us,
    }

    out_path = os.path.join("quality", "phase1_runs", "i3_hdram.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
