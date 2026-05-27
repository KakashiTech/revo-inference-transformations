from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple

import torch


def measure_latency_distribution(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
    warmup: int = 2,
    runs: int = 10,
) -> Dict[str, float]:
    latencies: List[float] = []
    tokens = 0
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            for t in texts:
                enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
                tokens += int(enc.input_ids.numel())
                _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)
        for _ in range(runs):
            t0 = time.perf_counter()
            for t in texts:
                enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
                _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)
            lat = time.perf_counter() - t0
            latencies.append(lat)
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "mean": 0.0, "std": 0.0, "cv": 0.0, "runs": 0.0}
    lat_sorted = sorted(latencies)
    p50 = lat_sorted[int(0.5 * (len(lat_sorted) - 1))]
    p90 = lat_sorted[int(0.9 * (len(lat_sorted) - 1))]
    mean = statistics.mean(latencies)
    std = statistics.pstdev(latencies)
    cv = (std / mean) if mean > 0 else 0.0
    return {"p50": p50, "p90": p90, "mean": mean, "std": std, "cv": cv, "runs": float(len(latencies))}
