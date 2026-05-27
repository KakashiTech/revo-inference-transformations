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
# Phase I
from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank
# Phase II
from revo.hora import replace_with_hora
from revo.holography import replace_with_holography, calibrate_holo_pinn
# Phase III
from revo.spectral import prune_model_spectral
from revo.phase_bus import replace_with_phase_bus, calibrate_phase_bus
from revo.reversible import replace_with_reversible, calibrate_reversible
from revo.wdm import replace_with_wdm
from revo.fft_kernel import replace_with_circulant
# Phase IV
from revo.fractal import replace_with_fractal
from revo.ephemeral import replace_with_ephemeral, calibrate_ephemeral
from revo.radix import radix_eval_nll
# Phase V
from revo.energy import measure_energy
from revo.latency_monitor import measure_latency_distribution
from revo.mlir_kernels import compile_model_guarded


def _gen_texts(n: int) -> List[str]:
    base = [
        "Explica brevemente el filtrado espectral y su impacto.",
        "Describe el uso de matrices circulantes en FFT.",
        "¿Qué aporta HoRA sobre una variedad hiperbólica?",
        "Resume el mapeo holográfico bulk→boundary→bulk.",
        "Define rango efectivo y energía espectral.",
        "¿Qué es un bus de fase natural?",
        "Explica el un-computing reversible.",
        "¿Qué es WDM y cómo paraleliza subcanales?",
        "¿Cómo funciona un gating efímero suave?",
        "Explica el caching tipo árbol de prefijos (Radix).",
    ]
    texts = []
    for i in range(n):
        texts.append(base[i % len(base)] + f" [#{i}]")
    return texts


def _stage_metrics(model, tok, texts, max_length):
    rss0 = measure_memory_rss()
    t0 = time.perf_counter()
    nll = evaluate_nll(model, tok, texts, max_length=max_length)
    dt = time.perf_counter() - t0
    rss1 = measure_memory_rss()
    return {
        "nll": nll,
        "eval_time_s": dt,
        "params": count_parameters(model),
        "rss_bytes": rss1,
        "rss_delta_bytes": int(rss1 - rss0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO Full Pipeline (Phases I→V) with multi-stage metrics")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--max-length", type=int, default=128)
    # Phase toggles
    ap.add_argument("--no-phase1", action="store_true")
    ap.add_argument("--no-phase2", action="store_true")
    ap.add_argument("--no-phase3", action="store_true")
    ap.add_argument("--no-phase4", action="store_true")
    ap.add_argument("--no-phase5", action="store_true")
    # Phase I params
    ap.add_argument("--p1-energy-keep", type=float, default=0.92)
    ap.add_argument("--p1-max-rank", type=int, default=None)
    ap.add_argument("--p1-max-rank-frac", type=float, default=0.25)
    ap.add_argument("--p1-calibrate", action="store_true")
    # Phase II params
    ap.add_argument("--p2-use-hora", action="store_true")
    ap.add_argument("--p2-use-holo", action="store_true")
    ap.add_argument("--p2-rank", type=int, default=4)
    ap.add_argument("--p2-alpha", type=float, default=8.0)
    ap.add_argument("--p2-c", type=float, default=0.05)
    ap.add_argument("--p2-boundary-dim", type=int, default=16)
    ap.add_argument("--p2-holo-alpha", type=float, default=0.2)
    ap.add_argument("--p2-pinn-steps", type=int, default=10)
    ap.add_argument("--p2-pinn-lambda", type=float, default=1.0)
    # Phase III params
    ap.add_argument("--p3-energy-keep", type=float, default=0.90)
    ap.add_argument("--p3-phase-steps", type=int, default=10)
    ap.add_argument("--p3-phase-lambda", type=float, default=1.0)
    ap.add_argument("--p3-rev-rank", type=int, default=2)
    ap.add_argument("--p3-rev-steps", type=int, default=10)
    ap.add_argument("--p3-rev-lambda", type=float, default=1.0)
    ap.add_argument("--p3-bands", type=int, default=2)
    # Phase IV params
    ap.add_argument("--p4-depth", type=int, default=2)
    ap.add_argument("--p4-alpha", type=float, default=0.5)
    ap.add_argument("--p4-epi-steps", type=int, default=10)
    ap.add_argument("--p4-epi-lambda", type=float, default=1.0)
    # Phase V params
    ap.add_argument("--p5-warmup", type=int, default=2)
    ap.add_argument("--p5-runs", type=int, default=5)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()
        texts = _gen_texts(args.prompts)

        # Baseline
        base = _stage_metrics(model, tok, texts, args.max_length)

        stages: Dict[str, Dict[str, float]] = {"baseline": base}

        # Phase I: Low-rank compression for generic 2D-weight modules (Conv1D/Linear)
        if not args.no_phase1:
            prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], max_rank=args.p1_max_rank)
            ranks = allocate_ranks_energy_with_caps(prof, energy_keep=args.p1_energy_keep, max_rank=args.p1_max_rank, max_rank_frac=args.p1_max_rank_frac)
            _ = replace_2d_modules_with_lowrank(model, ranks, calibrate=bool(args.p1_calibrate), calibrate_samples=256, seed=seed)
            stages["phase1"] = _stage_metrics(model, tok, texts, args.max_length)

        # Phase II: HoRA + Holography (optional)
        if not args.no_phase2:
            if args.p2_use_hora:
                try:
                    replace_with_hora(model, rank=args.p2_rank, alpha=args.p2_alpha, c=args.p2_c, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
                except Exception:
                    pass
            if args.p2_use_holo:
                try:
                    replace_with_holography(model, boundary_dim=args.p2_boundary_dim, alpha=args.p2_holo_alpha, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
                    calibrate_holo_pinn(model, tok, texts=texts, steps=args.p2_pinn_steps, lr=1e-2, lambda_phys=args.p2_pinn_lambda, max_length=args.max_length)
                except Exception:
                    pass
            stages["phase2"] = _stage_metrics(model, tok, texts, args.max_length)

        # Phase III: Spectral + PhaseBus + Reversible + WDM + Circulant
        if not args.no_phase3:
            prune_model_spectral(model, energy_keep=args.p3_energy_keep, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            replace_with_phase_bus(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            calibrate_phase_bus(model, tok, texts=texts, steps=args.p3_phase_steps, lr=5e-2, lambda_phys=args.p3_phase_lambda, max_length=args.max_length)
            replace_with_reversible(model, rank=args.p3_rev_rank, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            calibrate_reversible(model, tok, texts=texts, steps=args.p3_rev_steps, lr=5e-2, lambda_phys=args.p3_rev_lambda, max_length=args.max_length)
            replace_with_wdm(model, bands=args.p3_bands, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            replace_with_circulant(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            stages["phase3"] = _stage_metrics(model, tok, texts, args.max_length)

        # Phase IV: Fractal + Ephemeral + Radix (eval only)
        if not args.no_phase4:
            replace_with_fractal(model, depth=args.p4_depth, alpha=args.p4_alpha, name_patterns=["attn", "mlp", "c_proj"], skip_lm_head=True)
            replace_with_ephemeral(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            calibrate_ephemeral(model, tok, texts=texts, steps=args.p4_epi_steps, lr=5e-2, lambda_phys=args.p4_epi_lambda, max_length=args.max_length)
            stages["phase4"] = _stage_metrics(model, tok, texts, args.max_length)
            stages["phase4_radix_eval"] = radix_eval_nll(model, tok, texts, max_length=args.max_length)

        # Phase V: Energy + Latency (+ optional compile)
        if not args.no_phase5:
            lat = measure_latency_distribution(model, tok, texts, max_length=args.max_length, warmup=args.p5_warmup, runs=args.p5_runs)
            energy = measure_energy(model, tok, texts, max_length=args.max_length)
            stages["phase5_latency"] = lat
            stages["phase5_energy"] = {
                "total_flops": energy.total_flops,
                "dyn_energy_j": energy.dyn_energy_j,
                "landauer_lower_j": energy.landauer_lower_j,
                "latency_s": energy.latency_s,
                "tokens": energy.tokens,
            }
        all_results.append({"seed": seed, "stages": stages})

    # Aggregate simple summary across seeds for final model stage (nll/time/params/rss)
    def _agg(keys: List[str], stage_key: str) -> Dict[str, float]:
        vals: Dict[str, List[float]] = {k: [] for k in keys}
        for r in all_results:
            s = r["stages"].get(stage_key, {})
            for k in keys:
                v = s.get(k)
                if isinstance(v, (int, float)):
                    vals[k].append(float(v))
        return {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}

    final_stage = "phase5_energy" if not args.no_phase5 else ("phase4" if not args.no_phase4 else ("phase3" if not args.no_phase3 else ("phase2" if not args.no_phase2 else "phase1")))
    summary = {
        "seeds": seeds,
        "prompts": args.prompts,
        "final_stage": final_stage,
        "final_metrics_mean": _agg(["nll", "eval_time_s", "params", "rss_bytes"], stage_key=(final_stage if final_stage in ["baseline", "phase1", "phase2", "phase3", "phase4"] else "baseline")),
    }

    out_path = args.results_json
    if out_path is None:
        ts = int(time.time())
        out_dir = os.path.join("quality", "full_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"full_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_results, "summary": summary}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": summary, "results_path": out_path}, indent=2))


if __name__ == "__main__":
    main()
