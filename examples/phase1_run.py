from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import count_parameters, evaluate_nll, measure_memory_rss
from revo.layer_profile import (
    profile_model,
    allocate_ranks_by_energy,
    get_linear_layers,
    allocate_ranks_energy_with_caps,
)
from revo.lowrank import replace_linear_with_lowrank
from revo.archive.tensor_train import replace_linear_with_tt2


def _texts_default() -> List[str]:
    return [
        "Hola REVO! Resume en dos líneas qué es una descomposición de rango bajo.",
        "Explain in one sentence what spectral truncation does to a weight matrix.",
        "Lista tres ventajas de aproximar capas densas con TT/MPO.",
        "¿Por qué calibrar (healing) tras truncamiento SVD puede estabilizar precisión?",
        "Give a short definition of effective rank and its relation to information energy.",
    ]


def _collect_profile_stats(profile: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    entropies = []
    eff_ranks = []
    for _, p in profile.items():
        en = float(p.get("entropy_norm", 0.0))
        er = float(p.get("effective_rank", 0.0))
        entropies.append(en)
        eff_ranks.append(er)
    return {
        "entropy_norm_mean": float(np.mean(entropies) if entropies else 0.0),
        "effective_rank_mean": float(np.mean(eff_ranks) if eff_ranks else 0.0),
    }


def _compute_theoretical_savings(profile: Dict[str, Dict[str, object]], ranks: Dict[str, int], include_bias: bool = True, dtype_bytes: int = 4) -> Dict[str, float]:
    orig = 0
    comp = 0
    for name, p in profile.items():
        of = int(p.get("out_features", 0))
        inf = int(p.get("in_features", 0))
        if of <= 0 or inf <= 0:
            continue
        r = int(ranks.get(name, 0))
        if r <= 0:
            continue
        orig_params = of * inf + (of if include_bias else 0)
        comp_params = of * r + r * inf + (of if include_bias else 0)
        orig += orig_params
        comp += comp_params
    saved_params = max(0, orig - comp)
    ratio = (float(saved_params) / float(orig)) if orig > 0 else 0.0
    saved_mb = float(saved_params * dtype_bytes) / (1024.0 * 1024.0)
    return {"orig_params": float(orig), "comp_params": float(comp), "saved_params": float(saved_params), "saved_ratio": ratio, "saved_mb": saved_mb}


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO Phase I Pipeline: Profiling, Low-Rank/TT compression, Healing, Evaluation")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("REVO_SEED", "0") or 0))
    ap.add_argument("--energy-keep", type=float, default=0.98, help="Fraction of spectral energy to retain per layer")
    ap.add_argument("--max-rank", type=int, default=None)
    ap.add_argument("--max-rank-frac", type=float, default=0.25, help="Per-layer cap as fraction of min(out,in)")
    ap.add_argument("--method", choices=["lowrank", "tt2"], default="lowrank")
    ap.add_argument("--calibrate", action="store_true", help="Enable healing calibration for lowrank")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load model for baseline eval and profiling
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    # Baseline metrics
    rss_before = measure_memory_rss()
    t0 = time.perf_counter()
    base_nll = evaluate_nll(model, tok, _texts_default(), max_length=args.max_length)
    base_time = time.perf_counter() - t0
    base_params = count_parameters(model)

    # Profiling and rank allocation
    prof = profile_model(model, max_rank=args.max_rank)
    # Rank allocation with parameter-saving caps to avoid blow-up
    ranks = allocate_ranks_energy_with_caps(
        prof,
        energy_keep=args.energy_keep,
        max_rank=args.max_rank,
        max_rank_frac=args.max_rank_frac,
    )
    prof_stats = _collect_profile_stats(prof)

    # Compression
    if args.method == "lowrank":
        rep = replace_linear_with_lowrank(model, ranks, calibrate=args.calibrate, calibrate_samples=args.calib_samples, seed=args.seed)
    else:
        rep = replace_linear_with_tt2(model, ranks)

    comp_params = count_parameters(model)
    rss_after = measure_memory_rss()

    # Post-compression evaluation
    t1 = time.perf_counter()
    comp_nll = evaluate_nll(model, tok, _texts_default(), max_length=args.max_length)
    comp_time = time.perf_counter() - t1

    # Theoretical savings on targeted layers (float32 assumed)
    theo = _compute_theoretical_savings(prof, ranks, include_bias=True, dtype_bytes=4)

    results = {
        "model": args.model,
        "seed": args.seed,
        "method": args.method,
        "energy_keep": args.energy_keep,
        "max_rank": args.max_rank,
        "calibrate": bool(args.calibrate),
        "texts_n": len(_texts_default()),
        "baseline": {
            "nll": base_nll,
            "eval_time_s": base_time,
            "params": base_params,
            "rss_bytes": rss_before,
        },
        "compressed": {
            "nll": comp_nll,
            "eval_time_s": comp_time,
            "params": comp_params,
            "rss_bytes": rss_after,
        },
        "ranks": ranks,
        "profile_stats": prof_stats,
        "theoretical_savings": theo,
        "timestamp": int(time.time()),
    }

    # Save results
    out_path = args.results_json
    if out_path is None:
        ts = int(time.time())
        out_dir = os.path.join("quality", "phase1_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"phase1_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print short summary
    summary = {
        "nll_delta": float(comp_nll - base_nll),
        "eval_time_ratio": float(comp_time / max(1e-9, base_time)),
        "params_ratio": float(comp_params / max(1, base_params)),
        "rss_delta_bytes": int(rss_after - rss_before),
        "saved_ratio_targeted": float(theo.get("saved_ratio", 0.0)),
        "saved_mb_targeted": float(theo.get("saved_mb", 0.0)),
        "results_path": out_path,
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
