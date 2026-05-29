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
from revo.spectral import prune_model_spectral
from revo.fft_kernel import replace_with_circulant
from revo.phase_bus import replace_with_phase_bus, calibrate_phase_bus
from revo.reversible import replace_with_reversible, calibrate_reversible
from revo.wdm import replace_with_wdm
from revo.oscillatory_gating import OscillatoryHooks, interference_metrics, oscillatory_bptt_tune
from revo.archive.morse import morse_skeletonize
from revo.archive.mutual_information_fusion import mi_fuse_outputs
from revo.archive.low_dimensional import consolidate_lowdim


def _texts_default() -> List[str]:
    return [
        "Explica brevemente el filtrado espectral de pesos.",
        "Describe en una frase el uso de matrices circulantes para acelerar multiplicaciones.",
        "¿Qué impacto tiene mantener el 90% de energía espectral en precisión?",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO Phase III Pipeline: ONN (Osc), Interference, Phase sync + Spectral/Circulant/PhaseBus/Reversible/WDM")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("REVO_SEED", "0") or 0))
    # Oscillatory gating (ONN)
    ap.add_argument("--use-osc", action="store_true")
    ap.add_argument("--osc-alpha", type=float, default=0.05)
    ap.add_argument("--osc-freq", type=float, default=0.25)
    ap.add_argument("--osc-kappa", type=float, default=0.0, help="Kuramoto coupling strength")
    ap.add_argument("--osc-steps", type=int, default=0, help="Kuramoto steps to run before eval")
    ap.add_argument("--interf-dphi", type=float, default=0.25, help="Delta phase for interference metrics")
    # Oscillatory BPTT (phase tuning)
    ap.add_argument("--bptt-steps", type=int, default=0)
    ap.add_argument("--bptt-lr", type=float, default=5e-2)
    # Phase 2 consolidation (optional pre-ONN)
    ap.add_argument("--p2-use-morse", action="store_true")
    ap.add_argument("--p2-morse-keep-frac", type=float, default=0.995)
    ap.add_argument("--p2-use-mi", action="store_true")
    ap.add_argument("--p2-mi-threshold", type=float, default=0.995)
    ap.add_argument("--p2-mi-max-samples", type=int, default=1024)
    ap.add_argument("--p2-use-lowdim", action="store_true")
    ap.add_argument("--p2-lowdim-frac", type=float, default=0.95)
    ap.add_argument("--p2-lowdim-max-samples", type=int, default=512)
    ap.add_argument("--p2-lowdim-dtype", type=str, default="float16")
    ap.add_argument("--use-spectral", action="store_true")
    ap.add_argument("--energy-keep", type=float, default=0.90)
    ap.add_argument("--use-circulant", action="store_true")
    # Phase bus (natural frequency tuning)
    ap.add_argument("--use-phase-bus", action="store_true")
    ap.add_argument("--phase-steps", type=int, default=0)
    ap.add_argument("--phase-lambda", type=float, default=1.0)
    # Reversible un-compute
    ap.add_argument("--use-reversible", action="store_true")
    ap.add_argument("--rev-rank", type=int, default=2)
    ap.add_argument("--rev-steps", type=int, default=0)
    ap.add_argument("--rev-lambda", type=float, default=1.0)
    # WDM
    ap.add_argument("--use-wdm", action="store_true")
    ap.add_argument("--bands", type=int, default=2)
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

    spectral_report = {}
    circulant_report = {}
    phase_bus_report = {}
    reversible_report = {}
    wdm_report = {}
    osc_attached: List[str] = []
    osc_coherence: float = 0.0
    interf_cos: float = 1.0
    interf_kl: float = 0.0
    osc = None
    bptt_report: Dict[str, float] = {}
    p2_morse_report: Dict[str, Dict[str, int]] = {}
    p2_mi_report: Dict[str, Dict[str, int]] = {}
    p2_lowdim_report: Dict[str, Tuple[int, int, int]] = {}

    if args.use_spectral:
        spectral_report = prune_model_spectral(
            model,
            energy_keep=args.energy_keep,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )

    if args.use_phase_bus:
        phase_bus_report = replace_with_phase_bus(
            model,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )
        if args.phase_steps and args.phase_steps > 0:
            calibrate_phase_bus(
                model,
                tok,
                texts=_texts_default(),
                steps=args.phase_steps,
                lr=5e-2,
                lambda_phys=args.phase_lambda,
                max_length=args.max_length,
            )
    if args.use_reversible:
        reversible_report = replace_with_reversible(
            model,
            rank=args.rev_rank,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )
        if args.rev_steps and args.rev_steps > 0:
            calibrate_reversible(
                model,
                tok,
                texts=_texts_default(),
                steps=args.rev_steps,
                lr=5e-2,
                lambda_phys=args.rev_lambda,
                max_length=args.max_length,
            )
    if args.use_wdm:
        wdm_report = replace_with_wdm(
            model,
            bands=args.bands,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )
    if args.use_circulant:
        circulant_report = replace_with_circulant(
            model,
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )

    if args.use_osc:
        osc = OscillatoryHooks(alpha=float(args.osc_alpha), freq=float(args.osc_freq))
        osc_attached = osc.attach(model, name_patterns=patterns, skip_lm_head=bool(args.skip_lm_head))
        if args.osc_steps and args.osc_kappa != 0.0:
            osc.kuramoto(steps=int(args.osc_steps), kappa=float(args.osc_kappa), dt=0.1)
        osc_coherence = float(osc.phase_coherence())
        # Optional BPTT tuning on phases before final eval
        if int(args.bptt_steps) > 0:
            try:
                bptt_report = oscillatory_bptt_tune(
                    model,
                    tok,
                    texts=_texts_default(),
                    manager=osc,
                    steps=int(args.bptt_steps),
                    lr=float(args.bptt_lr),
                    kappa=float(args.osc_kappa),
                    dt=0.1,
                    max_length=args.max_length,
                )
            except Exception:
                bptt_report = {"steps": float(args.bptt_steps), "lr": float(args.bptt_lr), "error": 1.0}
            try:
                gc.collect()
            except Exception:
                pass
            free_memory_trim()

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
    if args.use_osc and osc is not None:
        try:
            interf_cos, interf_kl = interference_metrics(
                model, tok, _texts_default(), manager=osc, delta_phi=float(args.interf_dphi), max_length=args.max_length,
            )
        except Exception:
            pass
    try:
        gc.collect()
    except Exception:
        pass
    free_memory_trim()

    results = {
        "model": args.model,
        "seed": args.seed,
        "use_spectral": bool(args.use_spectral),
        "energy_keep": args.energy_keep,
        "use_circulant": bool(args.use_circulant),
        "patterns": patterns,
        "skip_lm_head": bool(args.skip_lm_head),
        "baseline": {
            "nll": base_nll,
            "eval_time_s": base_time,
            "params": base_params,
            "rss_bytes": rss_before,
        },
        "with_phase3": {
            "nll": comp_nll,
            "eval_time_s": comp_time,
            "params": comp_params,
            "rss_bytes": rss_after,
        },
        "p2_morse_report": p2_morse_report,
        "p2_mi_report": p2_mi_report,
        "p2_lowdim_report": p2_lowdim_report,
        "osc_report": {
            "enabled": bool(args.use_osc),
            "alpha": float(args.osc_alpha),
            "freq": float(args.osc_freq),
            "kappa": float(args.osc_kappa),
            "steps": int(args.osc_steps),
            "attached": osc_attached,
            "phase_coherence": osc_coherence,
            "interference_cos": interf_cos,
            "interference_sym_kl": interf_kl,
        },
        "bptt_report": bptt_report,
        "spectral_report": spectral_report,
        "circulant_report": circulant_report,
        "phase_bus_report": phase_bus_report,
        "reversible_report": reversible_report,
        "wdm_report": wdm_report,
        "timestamp": int(time.time()),
    }

    out_path = args.results_json
    if out_path is None:
        ts = int(time.time())
        out_dir = os.path.join("quality", "phase3_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"phase3_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = {
        "nll_delta": float(comp_nll - base_nll),
        "eval_time_ratio": float(comp_time / max(1e-9, base_time)),
        "params_delta": int(comp_params - base_params),
        "rss_delta_bytes": int(rss_after - rss_before),
        "results_path": out_path,
        "osc_layers": len(osc_attached) if osc_attached else 0,
        "bptt_steps": int(args.bptt_steps),
        "p2_morse_pruned_total": int(sum((v.get("pruned", 0) for v in p2_morse_report.values()))) if p2_morse_report else 0,
        "p2_mi_fused_pairs": int(sum((v.get("fused_pairs", 0) for v in p2_mi_report.values()))) if p2_mi_report else 0,
        "p2_lowdim_count": int(len(p2_lowdim_report)) if p2_lowdim_report else 0,
        "spectral_layers": len(spectral_report) if spectral_report else 0,
        "circulant_layers": len(circulant_report) if circulant_report else 0,
        "phase_bus_layers": len(phase_bus_report) if phase_bus_report else 0,
        "reversible_layers": len(reversible_report) if reversible_report else 0,
        "wdm_layers": len(wdm_report) if wdm_report else 0,
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
