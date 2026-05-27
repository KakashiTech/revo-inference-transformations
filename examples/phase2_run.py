from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import count_parameters, evaluate_nll, free_memory_trim, measure_memory_rss
from revo.hora import replace_with_hora
from revo.holography import replace_with_holography, calibrate_holo_pinn
from revo.morse import morse_skeletonize
from revo.mutual_information_fusion import mi_fuse_outputs
from revo.low_dimensional import consolidate_lowdim


def _texts_default() -> List[str]:
    return [
        "Resume en dos líneas qué es HoRA y por qué opera en espacios hiperbólicos.",
        "Explain why Poincaré models are suitable for hierarchical data.",
        "Da un ejemplo de adaptación low-rank aplicada en el espacio tangente.",
    ]


def _adapter_params_from_report(report: Dict[str, Tuple[int, int]], in_out_map: Optional[Dict[str, Tuple[int, int]]] = None) -> int:
    # report: name -> (out_features, rank)
    # If in_out_map available: name -> (in_features, out_features)
    total = 0
    for name, (out_f, r) in report.items():
        if in_out_map and name in in_out_map:
            in_f = int(in_out_map[name][0])
        else:
            # Fallback: assume in_features = out_features (rough upper bound)
            in_f = int(out_f)
        total += int(in_f * r + r * out_f)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO Phase II Pipeline: HoRA and Holography (bulk→boundary) with optional hyperbolic geometry")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("REVO_SEED", "0") or 0))
    # HoRA controls
    ap.add_argument("--use-hora", action="store_true", help="Apply HoRA adapters")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--c", type=float, default=0.0, help="Curvature for Poincaré (0.0 = Euclidean)")
    # Holography controls
    ap.add_argument("--use-holo", action="store_true", help="Apply Holography bulk→boundary adapters")
    ap.add_argument("--boundary-dim", type=int, default=16)
    ap.add_argument("--holo-alpha", type=float, default=0.2)
    ap.add_argument("--pinn-steps", type=int, default=0, help="If >0, run PINN-like calibration of holography gates")
    ap.add_argument("--pinn-lambda", type=float, default=1.0)
    # Topological Phase II controls
    ap.add_argument("--use-morse", action="store_true", help="Apply discrete-Morse inspired skeletonization by pruning least-influential outputs per module")
    ap.add_argument("--morse-keep-frac", type=float, default=0.995)
    ap.add_argument("--use-mi", action="store_true", help="Apply MI-based fusion tying highly correlated outputs per module")
    ap.add_argument("--mi-threshold", type=float, default=0.995)
    ap.add_argument("--mi-max-samples", type=int, default=1024)
    ap.add_argument("--use-lowdim", action="store_true", help="Consolidate modules projecting inputs to low-dimensional PCA subspace")
    ap.add_argument("--lowdim-frac", type=float, default=0.99)
    ap.add_argument("--lowdim-max-samples", type=int, default=1024)
    ap.add_argument("--lowdim-dtype", type=str, default="float32")
    # Selection and general
    ap.add_argument("--patterns", type=str, default="attn,mlp,c_fc,c_proj")
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
    report: Dict[str, Tuple[int, int]] = {}
    report_holo: Dict[str, Tuple[int, int, int]] = {}
    report_morse: Dict[str, Dict[str, int]] = {}
    report_mi: Dict[str, Dict[str, int]] = {}
    report_lowdim: Dict[str, Tuple[int, int, int]] = {}
    if args.use_hora:
        report = replace_with_hora(
            model,
            rank=args.rank,
            alpha=args.alpha,
            c=args.c,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )
    if args.use_holo:
        report_holo = replace_with_holography(
            model,
            boundary_dim=args.boundary_dim,
            alpha=args.holo_alpha,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )
        if args.pinn_steps and args.pinn_steps > 0:
            calibrate_holo_pinn(
                model,
                tok,
                texts=_texts_default(),
                steps=args.pinn_steps,
                lr=5e-2,
                lambda_phys=args.pinn_lambda,
                max_length=args.max_length,
            )
    if args.use_morse:
        report_morse = morse_skeletonize(
            model,
            keep_frac=float(args.morse_keep_frac),
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )
    if args.use_mi:
        report_mi = mi_fuse_outputs(
            model,
            tok,
            texts=_texts_default(),
            name_patterns=patterns,
            threshold=float(args.mi_threshold),
            max_length=args.max_length,
            max_samples=int(args.mi_max_samples),
        )
    if args.use_lowdim:
        report_lowdim = consolidate_lowdim(
            model,
            tok,
            texts=_texts_default(),
            name_patterns=patterns,
            frac=float(args.lowdim_frac),
            max_length=args.max_length,
            max_samples=int(args.lowdim_max_samples),
            dtype=str(args.lowdim_dtype),
        )

    # Post-application metrics
    comp_params = count_parameters(model)
    try:
        gc.collect()
    except Exception:
        pass
    free_memory_trim()
    rss_after = measure_memory_rss()
    t1 = time.perf_counter()
    comp_nll = evaluate_nll(model, tok, _texts_default(), max_length=args.max_length)
    comp_time = time.perf_counter() - t1
    try:
        gc.collect()
    except Exception:
        pass
    free_memory_trim()

    results = {
        "model": args.model,
        "seed": args.seed,
        "hora": {
            "enabled": bool(args.use_hora),
            "rank": args.rank,
            "alpha": args.alpha,
            "c": args.c,
        },
        "holography": {
            "enabled": bool(args.use_holo),
            "boundary_dim": args.boundary_dim,
            "alpha": args.holo_alpha,
            "pinn_steps": args.pinn_steps,
            "pinn_lambda": args.pinn_lambda,
        },
        "patterns": patterns,
        "skip_lm_head": bool(args.skip_lm_head),
        "baseline": {
            "nll": base_nll,
            "eval_time_s": base_time,
            "params": base_params,
            "rss_bytes": rss_before,
        },
        "with_phase2": {
            "nll": comp_nll,
            "eval_time_s": comp_time,
            "params": comp_params,
            "rss_bytes": rss_after,
        },
        "replaced_hora": report,
        "replaced_holo": report_holo,
        "morse": report_morse,
        "mi_fusion": report_mi,
        "lowdim": report_lowdim,
        "timestamp": int(time.time()),
    }

    out_path = args.results_json
    if out_path is None:
        ts = int(time.time())
        out_dir = os.path.join("quality", "phase2_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"phase2_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Aggregate simple counts for quick glance
    morse_pruned = int(sum((v.get("pruned", 0) for v in report_morse.values()))) if report_morse else 0
    mi_pairs = int(sum((v.get("fused_pairs", 0) for v in report_mi.values()))) if report_mi else 0
    lowdim_count = int(len(report_lowdim)) if report_lowdim else 0
    summary = {
        "nll_delta": float(comp_nll - base_nll),
        "eval_time_ratio": float(comp_time / max(1e-9, base_time)),
        "params_delta": int(comp_params - base_params),
        "rss_delta_bytes": int(rss_after - rss_before),
        "results_path": out_path,
        "hora_count": int(len(report)),
        "holo_count": int(len(report_holo)),
        "morse_pruned_total": morse_pruned,
        "mi_fused_pairs": mi_pairs,
        "lowdim_count": lowdim_count,
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
