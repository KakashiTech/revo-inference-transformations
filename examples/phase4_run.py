from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import count_parameters, evaluate_nll, free_memory_trim, measure_memory_rss
from revo.fractal import replace_with_fractal
from revo.ephemeral import replace_with_ephemeral, calibrate_ephemeral
from revo.radix import radix_eval_nll
from revo.archive.hyperbolic_gating import solomonoff_mixed_nll, compositional_consistency, hyperbolic_profile, mdl_surrogate_nll


def _texts_default() -> List[str]:
    return [
        "Resume en dos líneas qué aporta una composición fractal de capas lineales.",
        "Explain what a KV prefix-tree cache can save during causal inference.",
        "¿Cómo afecta un gating efímero suave a la estabilidad del NLL?",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO Phase IV Pipeline: Topos/Kolmogorov/Solomonoff + Fractal/Ephemeral/Radix")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("REVO_SEED", "0") or 0))
    # Fractal
    ap.add_argument("--use-fractal", action="store_true")
    ap.add_argument("--fractal-depth", type=int, default=2)
    ap.add_argument("--fractal-alpha", type=float, default=0.5)
    # Ephemeral gating
    ap.add_argument("--use-ephemeral", action="store_true")
    ap.add_argument("--epi-steps", type=int, default=0)
    ap.add_argument("--epi-lambda", type=float, default=1.0)
    # Radix cache
    ap.add_argument("--use-radix", action="store_true")
    # Phase 4 metrics
    ap.add_argument("--use-solomonoff", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--use-topos-proxy", action="store_true")
    ap.add_argument("--use-hyper-profile", action="store_true")
    ap.add_argument("--use-mdl", action="store_true")
    ap.add_argument("--mdl-lambda", type=float, default=0.01)
    # General
    ap.add_argument("--patterns", type=str, default="attn,mlp,c_proj")
    ap.add_argument("--skip-lm-head", action="store_true")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    # Baseline
    t0 = time.perf_counter()
    base_nll = evaluate_nll(model, tok, _texts_default(), max_length=args.max_length)
    base_time = time.perf_counter() - t0
    base_params = count_parameters(model)
    try:
        gc.collect()
    except Exception:
        pass
    free_memory_trim()
    rss_before = measure_memory_rss()

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]

    report_fractal = {}
    report_ephemeral = {}
    report_radix = {}

    if args.use_fractal:
        report_fractal = replace_with_fractal(
            model,
            depth=args.fractal_depth,
            alpha=args.fractal_alpha,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )

    if args.use_ephemeral:
        report_ephemeral = replace_with_ephemeral(
            model,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )
        if args.epi_steps and args.epi_steps > 0:
            calibrate_ephemeral(
                model,
                tok,
                texts=_texts_default(),
                steps=args.epi_steps,
                lr=5e-2,
                lambda_phys=args.epi_lambda,
                max_length=args.max_length,
            )

    comp_params = count_parameters(model)
    try:
        gc.collect()
    except Exception:
        pass
    free_memory_trim()
    rss_after = measure_memory_rss()

    # Post-mod evaluation (naive)
    t1 = time.perf_counter()
    comp_nll = evaluate_nll(model, tok, _texts_default(), max_length=args.max_length)
    comp_time = time.perf_counter() - t1

    # Radix evaluation
    if args.use_radix:
        report_radix = radix_eval_nll(model, tok, _texts_default(), max_length=args.max_length)

    # Phase 4 metrics
    phase4_metrics: Dict[str, object] = {}
    if args.use_solomonoff:
        try:
            phase4_metrics["solomonoff_nll"] = float(solomonoff_mixed_nll(model, tok, _texts_default(), gamma=float(args.gamma), max_length=args.max_length))
        except Exception:
            phase4_metrics["solomonoff_nll_error"] = True
    if args.use_topos_proxy:
        try:
            phase4_metrics["compositional_consistency"] = float(compositional_consistency(model, tok, _texts_default(), max_length=args.max_length))
        except Exception:
            phase4_metrics["compositional_consistency_error"] = True
    if args.use_hyper_profile:
        try:
            phase4_metrics["hyperbolic_profile"] = hyperbolic_profile(model, tok, _texts_default(), max_length=args.max_length)
        except Exception:
            phase4_metrics["hyperbolic_profile_error"] = True
    if args.use_mdl:
        try:
            phase4_metrics["mdl_surrogate_nll"] = float(mdl_surrogate_nll(model, tok, _texts_default(), mdl_lambda=float(args.mdl_lambda), max_length=args.max_length))
        except Exception:
            phase4_metrics["mdl_surrogate_nll_error"] = True

    results = {
        "model": args.model,
        "seed": args.seed,
        "fractal": {"enabled": bool(args.use_fractal), "depth": args.fractal_depth, "alpha": args.fractal_alpha},
        "ephemeral": {"enabled": bool(args.use_ephemeral), "epi_steps": args.epi_steps, "epi_lambda": args.epi_lambda},
        "radix": {"enabled": bool(args.use_radix)},
        "patterns": patterns,
        "skip_lm_head": bool(args.skip_lm_head),
        "baseline": {
            "nll": base_nll,
            "eval_time_s": base_time,
            "params": base_params,
            "rss_bytes": rss_before,
        },
        "with_phase4": {
            "nll": comp_nll,
            "eval_time_s": comp_time,
            "params": comp_params,
            "rss_bytes": rss_after,
        },
        "report_fractal": report_fractal,
        "report_ephemeral": report_ephemeral,
        "report_radix": report_radix,
        "timestamp": int(time.time()),
        "phase4_metrics": phase4_metrics,
    }

    out_path = args.results_json
    if out_path is None:
        ts = int(time.time())
        out_dir = os.path.join("quality", "phase4_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"phase4_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = {
        "nll_delta": float(comp_nll - base_nll),
        "eval_time_ratio": float(comp_time / max(1e-9, base_time)),
        "params_delta": int(comp_params - base_params),
        "rss_delta_bytes": int(rss_after - rss_before),
        "results_path": out_path,
        "fractal_layers": len(report_fractal) if report_fractal else 0,
        "ephemeral_layers": len(report_ephemeral) if report_ephemeral else 0,
        "radix_saved_calls_ratio": float(report_radix.get("saved_calls_ratio", 0.0)) if report_radix else 0.0,
        "solomonoff_nll": float(phase4_metrics.get("solomonoff_nll", 0.0)) if phase4_metrics else 0.0,
        "comp_consistency": float(phase4_metrics.get("compositional_consistency", 0.0)) if phase4_metrics else 0.0,
        "hyper_over_euclid": float(phase4_metrics.get("hyperbolic_profile", {}).get("hyper_over_euclid", 0.0)) if isinstance(phase4_metrics.get("hyperbolic_profile"), dict) else 0.0,
        "mdl_surrogate_nll": float(phase4_metrics.get("mdl_surrogate_nll", 0.0)) if phase4_metrics else 0.0,
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
