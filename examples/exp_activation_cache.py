"""ActivationCache — benchmark de velocidad con re-ejecución parcial.

Compara:
  - Full forward (sin cache): re-ejecuta todas las 12 capas
  - Partial forward (con cache): re-ejecuta solo desde primera capa modificada
  - lm_head-only (sin re-ejecución): solo lm_head
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revo._logging import get_logger
from revo.activation_cache import ActivationCache
from revo.field_selector import FieldSelector, FieldSelectorConfig

log = get_logger(__name__)

TEXTS = [
    "The capital of France is Paris and it has been for centuries a center of art",
    "Quantum computing relies on superposition and entanglement to solve complex problems",
    "DNA contains the genetic instructions for the development and functioning of all life",
    "Neural networks are composed of layers of interconnected nodes that process information",
]


@torch.no_grad()
def benchmark_full_vs_partial(model, tokenizer, texts, n_trials=3):
    cache = ActivationCache()
    times: dict = {"full": [], "partial": [], "lm_head_only": []}

    for text in texts:
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        ids = inp.input_ids
        slen = ids.shape[1]
        if slen < 2:
            continue

        for pos in range(slen - 1):
            for trial in range(n_trials):
                # Full forward
                t0 = time.perf_counter()
                logits_full, layers = cache.cache_forward(model, ids)
                # Simulate modifying layer 8
                old_w = model.transformer.h[8].mlp.c_proj.weight.data.clone()
                model.transformer.h[8].mlp.c_proj.weight.data += 0.001
                logits_full2 = cache.forward_from(model, start_layer=0)
                t_full = time.perf_counter() - t0
                model.transformer.h[8].mlp.c_proj.weight.data = old_w
                times["full"].append(t_full)

                # Partial forward (from layer 8)
                t0 = time.perf_counter()
                logits_partial, layers = cache.cache_forward(model, ids)
                old_w = model.transformer.h[8].mlp.c_proj.weight.data.clone()
                model.transformer.h[8].mlp.c_proj.weight.data += 0.001
                logits_partial2 = cache.forward_from(model, start_layer=8)
                t_partial = time.perf_counter() - t0
                model.transformer.h[8].mlp.c_proj.weight.data = old_w
                times["partial"].append(t_partial)

                # lm_head only (no re-run)
                t0 = time.perf_counter()
                logits_head, layers = cache.cache_forward(model, ids)
                hs = layers[-1]
                hs = model.transformer.ln_f(hs)
                logits_head2 = model.lm_head(hs)
                t_head = time.perf_counter() - t0
                times["lm_head_only"].append(t_head)

    def avg(lst):
        return float(np.mean(lst)) * 1000  # ms

    summary = {
        "n_measurements": len(times["full"]),
        "full_forward_ms": avg(times["full"]),
        "partial_forward_ms": avg(times["partial"]),
        "lm_head_only_ms": avg(times["lm_head_only"]),
        "speedup_partial_vs_full": avg(times["full"]) / max(avg(times["partial"]), 0.001),
        "speedup_head_vs_full": avg(times["full"]) / max(avg(times["lm_head_only"]), 0.001),
    }
    return summary


@torch.no_grad()
def benchmark_layermod_speed(model, tokenizer, texts):
    """Mide velocidad de step_full con y sin activation cache."""
    from revo.ephemeral_engine import EphemeralConfig, EphemeralEngine

    cfg = EphemeralConfig(
        use_codebook=False, use_field_selector=True,
        field_max_active=4, field_threshold=-0.5,
        novelty_threshold=99.0, delta_window_threshold=99.0,
        scale_max=0.8,
    )
    engine = EphemeralEngine(model, cfg)

    times_cached = []
    times_uncached = []

    for text in texts:
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
        ids = inp.input_ids
        slen = ids.shape[1]
        if slen < 2:
            continue

        for pos in range(slen - 1):
            t0 = time.perf_counter()
            logits, meta = engine.step_full(ids, pos, use_cache=True)
            dt = (time.perf_counter() - t0) * 1000
            if logits is not None:
                times_cached.append(dt)

            t0 = time.perf_counter()
            logits, meta = engine.step_full(ids, pos, use_cache=False)
            dt = (time.perf_counter() - t0) * 1000
            if logits is not None:
                times_uncached.append(dt)

    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    summary = {
        "n_measurements": len(times_cached),
        "cached_ms": avg(times_cached),
        "uncached_ms": avg(times_uncached),
        "speedup": avg(times_uncached) / max(avg(times_cached), 0.001),
    }
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ActivationCache Benchmark")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--results-json", default="quality/activation_cache/benchmark.json")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model: {args.model}")

    # Benchmark 1: raw forward passes
    print("\n--- Benchmark 1: Raw forward pass speed ---")
    r1 = benchmark_full_vs_partial(model, tokenizer, TEXTS, n_trials=2)
    print(f"  Full forward (12 layers):    {r1['full_forward_ms']:.1f}ms")
    print(f"  Partial forward (layers 8-11): {r1['partial_forward_ms']:.1f}ms")
    print(f"  lm_head only:                 {r1['lm_head_only_ms']:.1f}ms")
    print(f"  Speedup partial/full:         {r1['speedup_partial_vs_full']:.1f}x")
    print(f"  Speedup head/full:            {r1['speedup_head_vs_full']:.1f}x")

    # Benchmark 2: step_full with and without cache
    print("\n--- Benchmark 2: step_full() cached vs uncached ---")
    r2 = benchmark_layermod_speed(model, tokenizer, TEXTS)
    print(f"  Cached (rerun from first mod layer):   {r2['cached_ms']:.1f}ms")
    print(f"  Uncached (full rerun from scratch):    {r2['uncached_ms']:.1f}ms")
    print(f"  Speedup:                               {r2['speedup']:.1f}x")

    # Combined results
    combined = {
        "model": args.model,
        "forward_benchmark": r1,
        "step_full_benchmark": r2,
    }
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info("Saved to %s", args.results_json)
