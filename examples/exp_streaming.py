#!/usr/bin/env python3
"""Experiment: weight streaming vs full-load inference.

Measures:
  - NLL equality (streaming must be bit-exact)
  - Theoretical weight memory savings
  - Measured RSS peak comparison
  - Time overhead
  - Scaling with model size

Usage:
    python examples/exp_streaming.py --model distilgpt2
    python examples/exp_streaming.py --model Qwen/Qwen2.5-0.5B
    python examples/exp_streaming.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revo._utils import load_model_tokenizer, gen_texts, save_results_json, free_memory_trim
from revo.streaming import (
    shard_model, compare_memory, stream_forward, stream_generate,
    _free_all_block_weights, _restore_all_blocks,
)


def run_single_experiment(model_id: str, prompts: int = 20,
                           max_length: int = 128,
                           shard_dir: str = "/tmp/revo_shards",
                           generate: bool = True) -> Dict[str, Any]:
    """Run full vs streaming comparison for a single model.

    Returns a comprehensive report dict.
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    model, tokenizer = load_model_tokenizer(model_id)
    load_time = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n_params/1e6:.1f}M params in {load_time:.1f}s")

    shard_model(model, shard_dir)

    texts = gen_texts(prompts)
    from revo.streaming import _detect_arch
    arch = _detect_arch(model)
    report = {"model": model_id, "n_params": n_params, "arch": arch}

    # Phase 1: Comparison
    print(f"\n── Comparison (prompts={prompts}, max_len={max_length}) ──")
    comp = compare_memory(model, tokenizer, shard_dir, texts, max_length)
    report["comparison"] = comp
    t = comp["theory"]
    s = comp["streaming"]
    f = comp["full"]
    print(f"  NLL full:   {f['nll']:.8f}")
    print(f"  NLL stream: {s['nll']:.8f}  (Δ={s['nll_delta']:+.8f})")
    print(f"  Weight savings: {t['savings_pct']:.1f}%  (full={t['full_weights_mb']:.0f}MB → peak={t['stream_peak_weights_mb']:.0f}MB)")
    print(f"  RSS peak Δ:     {s['rss_peak_delta_mb']:.1f} MB")
    print(f"  Time ratio:     {s['time_ratio']:.1f}×")

    # Phase 2: Generate with streaming
    if generate:
        print(f"\n── Streaming generation ──")
        prompt = "The fundamental nature of consciousness is"
        gen_text_out, gen_meta = stream_generate(
            model, tokenizer, prompt, shard_dir,
            max_new_tokens=10, temperature=0.9, top_k=50, verbose=True,
        )
        report["generation"] = {
            "prompt": prompt,
            "output": gen_text_out,
            "meta": gen_meta,
        }
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {gen_text_out}")
        print(f"  Time:   {gen_meta['total_time_s']:.2f}s ({gen_meta['mean_time_per_token_s']*1000:.0f}ms/tok)")

    # Phase 3: Memory timeline
    print(f"\n── Memory timeline (1 layer at a time) ──")
    _free_all_block_weights(model)
    free_memory_trim()
    import psutil
    proc = __import__("psutil").Process(os.getpid())
    baseline_rss = proc.memory_info().rss
    print(f"  Bare RSS:  {baseline_rss/1024/1024:.0f} MB (no block weights)")

    from revo.streaming import _load_block_from_shard, _get_layer_container
    container = _get_layer_container(model)
    n_layers = len(container)
    layer_sizes = []
    for i in range(n_layers):
        _load_block_from_shard(
            container[i], shard_dir, i,
            next(model.parameters()).device,
            arch=comp.get("arch", "gpt2"),
        )
        rss = proc.memory_info().rss
        added = rss - baseline_rss
        layer_sizes.append(added)
        print(f"  Layer {i}: +{added/1024/1024:.1f} MB → {rss/1024/1024:.0f} MB total")
        for p in container[i].parameters():
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        free_memory_trim()

    report["memory_timeline"] = {
        "bare_rss_mb": baseline_rss / 1024 / 1024,
        "layer_rss_deltas_mb": [l / 1024 / 1024 for l in layer_sizes],
    }

    _restore_all_blocks(model, shard_dir)

    elapsed = time.perf_counter() - t0
    report["total_time_s"] = elapsed
    print(f"\nTotal experiment time: {elapsed:.1f}s")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Streaming weight experiment")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--all", action="store_true",
                    help="Run on multiple models")
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--shard-dir", default="/tmp/revo_shards")
    ap.add_argument("--output", default=None,
                    help="Output JSON path")
    ap.add_argument("--no-generate", action="store_true",
                    help="Skip generation phase")
    args = ap.parse_args()

    models = [
        "sshleifer/tiny-gpt2",
        "distilgpt2",
    ]
    if args.all:
        models += [
            "gpt2",
            "Qwen/Qwen2.5-0.5B",
        ]
    elif args.model:
        models = [args.model]

    all_reports = {}
    for mid in models:
        try:
            report = run_single_experiment(
                mid, prompts=args.prompts,
                max_length=args.max_length,
                shard_dir=os.path.join(args.shard_dir, mid.replace("/", "_")),
                generate=not args.no_generate,
            )
            all_reports[mid] = report
        except Exception as e:
            print(f"\nERROR on {mid}: {e}")
            import traceback
            traceback.print_exc()
            all_reports[mid] = {"error": str(e)}

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'N_layers':<10} {'Params(M)':<12} {'Wt_save%':<12} {'NLL Δ':<12} {'Time×':<10}")
    print(f"{'-'*70}")
    for mid, r in all_reports.items():
        if "error" in r:
            print(f"{mid:<30} ERROR: {r['error']}")
            continue
        c = r.get("comparison", {})
        t = c.get("theory", {})
        s = c.get("streaming", {})
        m = c.get("model", {})
        model_short = mid.split("/")[-1] if "/" in mid else mid
        print(f"{model_short:<30} {t.get('n_layers','?'):<10} {m.get('total_params',0)/1e6:<12.1f} {t.get('savings_pct',0):<11.1f}% {s.get('nll_delta',0):<+11.8f} {s.get('time_ratio',0):<9.1f}×")

    path = save_results_json(all_reports, default_dir="quality",
                             prefix="streaming_exp")
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
