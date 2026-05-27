from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Phase I
from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank
# Phase II (custom Phase II modules and protections)
from revo.hora import replace_with_hora
from revo.holography import replace_with_holography, calibrate_holo_pinn
from revo.beds import evaluate_beds, BEDSConfig
from revo.tqft import evaluate_tqft, TQFTConfig
from revo.category import evaluate_category
from revo.radix_cache import evaluate_radix, RadixConfig
from revo.frequency import evaluate_frequency, FreqConfig
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
from revo.mei_sync import measure_latency_distribution
from revo.mlir_kernels import compile_model_guarded
# Final Phases (VI–IX)
from revo.regimes import evaluate_regimes, RegimeConfig
from revo.probcal import evaluate_probcal, ProbCalConfig
from revo.implicit import evaluate_implicit, ImplicitConfig
from revo.biocomp import evaluate_biocomp, BioCompConfig


def _measure_rss_bytes() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


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
    return [base[i % len(base)] + f" [#{i}]" for i in range(n)]


def _eval_nll(model, tok, texts: List[str], max_length: int = 128) -> Tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc.input_ids
            attn = enc.attention_mask
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            loss = float(out.loss.item())
            n_tok = int(input_ids.numel())
            total_loss += loss * n_tok
            total_tokens += n_tok
    mean_nll = float(total_loss / max(1, total_tokens))
    return mean_nll, total_tokens


def _param_count(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _stage_metrics(model, tok, texts, max_length):
    rss0 = _measure_rss_bytes()
    t0 = time.perf_counter()
    nll, _ = _eval_nll(model, tok, texts, max_length=max_length)
    dt = time.perf_counter() - t0
    rss1 = _measure_rss_bytes()
    return {
        "nll": nll,
        "eval_time_s": dt,
        "params": _param_count(model),
        "rss_bytes": rss1,
        "rss_delta_bytes": int(rss1 - rss0),
    }


def cmd_pipeline(args: argparse.Namespace) -> None:
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
        if args.compile:
            model = compile_model_guarded(model)
        texts = _gen_texts(args.prompts)

        # Baseline
        base = _stage_metrics(model, tok, texts, args.max_length)
        stages: Dict[str, Dict[str, float]] = {"baseline": base}

        # Phase I
        if not args.no_phase1:
            prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], max_rank=args.p1_max_rank)
            ranks = allocate_ranks_energy_with_caps(prof, energy_keep=args.p1_energy_keep, max_rank=args.p1_max_rank, max_rank_frac=args.p1_max_rank_frac)
            _ = replace_2d_modules_with_lowrank(model, ranks, calibrate=bool(args.p1_calibrate), calibrate_samples=256, seed=seed)
            stages["phase1"] = _stage_metrics(model, tok, texts, args.max_length)

        # Phase II
        if not args.no_phase2:
            if args.p2_use_hora:
                try:
                    replace_with_hora(model, rank=args.p2_rank, alpha=args.p2_alpha, c=args.p2_c, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
                except Exception as e:
                    print(f"[WARN] Phase II HoRA failed: {e}")
            if args.p2_use_holo:
                try:
                    replace_with_holography(model, boundary_dim=args.p2_boundary_dim, alpha=args.p2_holo_alpha, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
                    calibrate_holo_pinn(model, tok, texts=texts, steps=args.p2_pinn_steps, lr=1e-2, lambda_phys=args.p2_pinn_lambda, max_length=args.max_length)
                except Exception as e:
                    print(f"[WARN] Phase II Holography failed: {e}")
            stages["phase2"] = _stage_metrics(model, tok, texts, args.max_length)

        # Phase III
        if not args.no_phase3:
            prune_model_spectral(model, energy_keep=args.p3_energy_keep, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            replace_with_phase_bus(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            calibrate_phase_bus(model, tok, texts=texts, steps=args.p3_phase_steps, lr=5e-2, lambda_phys=args.p3_phase_lambda, max_length=args.max_length)
            replace_with_reversible(model, rank=args.p3_rev_rank, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            calibrate_reversible(model, tok, texts=texts, steps=args.p3_rev_steps, lr=5e-2, lambda_phys=args.p3_rev_lambda, max_length=args.max_length)
            replace_with_wdm(model, bands=args.p3_bands, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            replace_with_circulant(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            stages["phase3"] = _stage_metrics(model, tok, texts, args.max_length)

        # Phase IV
        if not args.no_phase4:
            replace_with_fractal(model, depth=args.p4_depth, alpha=args.p4_alpha, name_patterns=["attn", "mlp", "c_proj"], skip_lm_head=True)
            replace_with_ephemeral(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], skip_lm_head=True)
            calibrate_ephemeral(model, tok, texts=texts, steps=args.p4_epi_steps, lr=5e-2, lambda_phys=args.p4_epi_lambda, max_length=args.max_length)
            stages["phase4"] = _stage_metrics(model, tok, texts, args.max_length)
            stages["phase4_radix_eval"] = radix_eval_nll(model, tok, texts, max_length=args.max_length)

        # Phase V
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

    def _agg(keys: List[str], stage_key: str) -> Dict[str, float]:
        vals: Dict[str, List[float]] = {k: [] for k in keys}
        for r in all_results:
            s = r["stages"].get(stage_key, {})
            for k in keys:
                v = s.get(k)
                if isinstance(v, (int, float)):
                    vals[k].append(float(v))
        return {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}

    final_stage = (
        "phase5_energy"
        if not args.no_phase5
        else (
            "phase4"
            if not args.no_phase4
            else ("phase3" if not args.no_phase3 else ("phase2" if not args.no_phase2 else "phase1"))
        )
    )
    summary = {
        "seeds": seeds,
        "prompts": args.prompts,
        "final_stage": final_stage,
        "final_metrics_mean": _agg(["nll", "eval_time_s", "params", "rss_bytes"], stage_key=(final_stage if final_stage in ["baseline", "phase1", "phase2", "phase3", "phase4", "phase5_energy", "phase5_latency"] else "baseline")),
    }

    out_path = args.results_json or os.path.join("quality", "full_runs", f"full_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_results, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary, "results_path": out_path}, indent=2))


def cmd_phase2_all(args: argparse.Namespace) -> None:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # BEDS
    beds_cfg = BEDSConfig(
        target_entropy=float(args.beds_target_entropy),
        ema_alpha=float(args.beds_ema_alpha),
        temp_min=float(args.beds_temp_min),
        temp_max=float(args.beds_temp_max),
        k_update=float(args.beds_k_update),
    )
    beds_res = evaluate_beds(args.model, seeds, int(args.prompts), int(args.max_length), beds_cfg)

    # TQFT
    tqft_cfg = TQFTConfig(topk=int(args.tqft_topk), num_braids=int(args.tqft_num_braids), noise_sigma=float(args.tqft_noise_sigma), seed=int(args.tqft_seed))
    tqft_res = evaluate_tqft(args.model, seeds, int(args.prompts), int(args.max_length), tqft_cfg)

    # Category
    cat_res = evaluate_category(args.model, seeds, int(args.prompts), int(args.max_length))

    # Radix (per-seed)
    radix_cfg = RadixConfig(prefix_bits=int(args.radix_prefix_bits), prefix_len=int(args.radix_prefix_len), bank_frac=float(args.radix_bank_frac))
    radix_variants = []
    for s in seeds:
        rr = evaluate_radix(args.model, int(s), int(args.prompts), int(args.max_length), radix_cfg)
        radix_variants.append(rr)
    radix_res = {
        "model": args.model,
        "seeds": seeds,
        "prompts": int(args.prompts),
        "max_length": int(args.max_length),
        "radix_config": {
            "prefix_bits": radix_cfg.prefix_bits,
            "prefix_len": radix_cfg.prefix_len,
            "bank_frac": radix_cfg.bank_frac,
        },
        "variants": radix_variants,
    }

    # Frequency
    freq_cfg = FreqConfig()
    freq_cfg.phases = int(args.freq_phases)
    freq_res = evaluate_frequency(args.model, seeds, int(args.prompts), int(args.max_length), freq_cfg)

    payload = {
        "model": args.model,
        "seeds": seeds,
        "prompts": int(args.prompts),
        "max_length": int(args.max_length),
        "beds": beds_res,
        "tqft": tqft_res,
        "category": cat_res,
        "radix": radix_res,
        "frequency": freq_res,
        "created_at": int(time.time()),
    }

    out_path = args.results_json or os.path.join("quality", "phase2_runs", f"phase2_all_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results_path": out_path}, indent=2))


def cmd_phase_final(args: argparse.Namespace) -> None:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Phase VI — Regímenes de existencia (micro↔macro)
    reg_cfg = RegimeConfig(
        density_weight=float(args.vi_density_weight),
        depth_weight=float(args.vi_depth_weight),
        size_weight=float(args.vi_size_weight),
        center=float(args.vi_center),
        sharpness=float(args.vi_sharpness),
        topk_entropy=int(args.vi_topk_entropy),
    )
    vi_res = evaluate_regimes(
        model_name=args.model,
        seeds=seeds,
        prompts=int(args.prompts),
        max_len=int(args.max_length),
        config=reg_cfg,
    )

    # Phase VII — Calibración probabilística profunda
    pc_cfg = ProbCalConfig(
        calib_frac=float(args.vii_calib_frac),
        steps=int(args.vii_steps),
        lr=float(args.vii_lr),
        alpha_curv=float(args.vii_alpha_curv),
        topk_eval=int(args.vii_topk_eval),
    )
    vii_res = evaluate_probcal(
        model_name=args.model,
        seeds=seeds,
        prompts=int(args.prompts),
        max_len=int(args.max_length),
        config=pc_cfg,
    )

    # Phase VIII — Existencia implícita (resonancia/activación local)
    imp_cfg = ImplicitConfig(
        codebook_k=int(args.viii_codebook_k),
        iters=int(args.viii_iters),
        q_quantile=float(args.viii_q_quantile),
    )
    viii_res = evaluate_implicit(
        model_name=args.model,
        seeds=seeds,
        prompts=int(args.prompts),
        max_len=int(args.max_length),
        config=imp_cfg,
    )

    # Phase IX — Convergencia bio-computacional
    bio_cfg = BioCompConfig(
        act_quantile=float(args.ix_act_quantile),
        topk_eval=int(args.ix_topk_eval),
        energy_warmup=int(args.ix_energy_warmup),
        energy_runs=int(args.ix_energy_runs),
    )
    ix_res = evaluate_biocomp(
        model_name=args.model,
        seeds=seeds,
        prompts=int(args.prompts),
        max_len=int(args.max_length),
        config=bio_cfg,
    )

    payload = {
        "model": args.model,
        "seeds": seeds,
        "prompts": int(args.prompts),
        "max_length": int(args.max_length),
        "phase_vi_regimes": vi_res,
        "phase_vii_probcal": vii_res,
        "phase_viii_implicit": viii_res,
        "phase_ix_biocomp": ix_res,
        "created_at": int(time.time()),
    }

    out_path = args.results_json or os.path.join("quality", "final_runs", f"final_all_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results_path": out_path}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO Unified Main CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Pipeline subcommand (Phases I→V)
    ap_p = sub.add_parser("pipeline", help="Run unified pipeline across phases I→V (toggle phases)")
    ap_p.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap_p.add_argument("--seeds", type=str, default="0")
    ap_p.add_argument("--prompts", type=int, default=50)
    ap_p.add_argument("--max-length", type=int, default=128)
    ap_p.add_argument("--compile", action="store_true")
    # Phase toggles
    ap_p.add_argument("--no-phase1", action="store_true")
    ap_p.add_argument("--no-phase2", action="store_true")
    ap_p.add_argument("--no-phase3", action="store_true")
    ap_p.add_argument("--no-phase4", action="store_true")
    ap_p.add_argument("--no-phase5", action="store_true")
    # Phase I params
    ap_p.add_argument("--p1-energy-keep", type=float, default=0.92)
    ap_p.add_argument("--p1-max-rank", type=int, default=None)
    ap_p.add_argument("--p1-max-rank-frac", type=float, default=0.25)
    ap_p.add_argument("--p1-calibrate", action="store_true")
    # Phase II params
    ap_p.add_argument("--p2-use-hora", action="store_true")
    ap_p.add_argument("--p2-use-holo", action="store_true")
    ap_p.add_argument("--p2-rank", type=int, default=4)
    ap_p.add_argument("--p2-alpha", type=float, default=8.0)
    ap_p.add_argument("--p2-c", type=float, default=0.05)
    ap_p.add_argument("--p2-boundary-dim", type=int, default=16)
    ap_p.add_argument("--p2-holo-alpha", type=float, default=0.2)
    ap_p.add_argument("--p2-pinn-steps", type=int, default=10)
    ap_p.add_argument("--p2-pinn-lambda", type=float, default=1.0)
    # Phase III params
    ap_p.add_argument("--p3-energy-keep", type=float, default=0.90)
    ap_p.add_argument("--p3-phase-steps", type=int, default=10)
    ap_p.add_argument("--p3-phase-lambda", type=float, default=1.0)
    ap_p.add_argument("--p3-rev-rank", type=int, default=2)
    ap_p.add_argument("--p3-rev-steps", type=int, default=10)
    ap_p.add_argument("--p3-rev-lambda", type=float, default=1.0)
    ap_p.add_argument("--p3-bands", type=int, default=2)
    # Phase IV params
    ap_p.add_argument("--p4-depth", type=int, default=2)
    ap_p.add_argument("--p4-alpha", type=float, default=0.5)
    ap_p.add_argument("--p4-epi-steps", type=int, default=10)
    ap_p.add_argument("--p4-epi-lambda", type=float, default=1.0)
    # Phase V params
    ap_p.add_argument("--p5-warmup", type=int, default=2)
    ap_p.add_argument("--p5-runs", type=int, default=5)
    ap_p.add_argument("--results-json", default=None)
    ap_p.set_defaults(func=cmd_pipeline)

    # Phase2-all subcommand (runs II.1–II.5 aggregating results)
    ap_a = sub.add_parser("phase2_all", help="Run all Phase II modules (II.1–II.5) and aggregate")
    ap_a.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap_a.add_argument("--seeds", type=str, default="0,1,2")
    ap_a.add_argument("--prompts", type=int, default=100)
    ap_a.add_argument("--max-length", type=int, default=128)
    # BEDS
    ap_a.add_argument("--beds-target-entropy", type=float, default=3.0)
    ap_a.add_argument("--beds-ema-alpha", type=float, default=0.1)
    ap_a.add_argument("--beds-temp-min", type=float, default=0.5)
    ap_a.add_argument("--beds-temp-max", type=float, default=2.0)
    ap_a.add_argument("--beds-k-update", type=float, default=0.1)
    # TQFT
    ap_a.add_argument("--tqft-topk", type=int, default=64)
    ap_a.add_argument("--tqft-num-braids", type=int, default=5)
    ap_a.add_argument("--tqft-noise-sigma", type=float, default=0.05)
    ap_a.add_argument("--tqft-seed", type=int, default=0)
    # Radix
    ap_a.add_argument("--radix-prefix-bits", type=int, default=64)
    ap_a.add_argument("--radix-prefix-len", type=int, default=16)
    ap_a.add_argument("--radix-bank-frac", type=float, default=0.5)
    # Frequency
    ap_a.add_argument("--freq-phases", type=int, default=8)
    ap_a.add_argument("--results-json", default=None)
    ap_a.set_defaults(func=cmd_phase2_all)

    # Final phases (VI–IX) subcommand
    ap_f = sub.add_parser("final_all", help="Run Final Phases VI–IX and aggregate results")
    ap_f.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap_f.add_argument("--seeds", type=str, default="0,1,2")
    ap_f.add_argument("--prompts", type=int, default=100)
    ap_f.add_argument("--max-length", type=int, default=128)
    ap_f.add_argument("--results-json", default=None)
    # VI — Regímenes
    ap_f.add_argument("--vi-density-weight", type=float, default=0.5)
    ap_f.add_argument("--vi-depth-weight", type=float, default=0.5)
    ap_f.add_argument("--vi-size-weight", type=float, default=0.25)
    ap_f.add_argument("--vi-center", type=float, default=0.6)
    ap_f.add_argument("--vi-sharpness", type=float, default=4.0)
    ap_f.add_argument("--vi-topk-entropy", type=int, default=32)
    # VII — ProbCal
    ap_f.add_argument("--vii-calib-frac", type=float, default=0.5)
    ap_f.add_argument("--vii-steps", type=int, default=50)
    ap_f.add_argument("--vii-lr", type=float, default=0.05)
    ap_f.add_argument("--vii-alpha-curv", type=float, default=0.1)
    ap_f.add_argument("--vii-topk-eval", type=int, default=64)
    # VIII — Implícito
    ap_f.add_argument("--viii-codebook-k", type=int, default=16)
    ap_f.add_argument("--viii-iters", type=int, default=5)
    ap_f.add_argument("--viii-q-quantile", type=float, default=0.5)
    # IX — BioComp
    ap_f.add_argument("--ix-act-quantile", type=float, default=0.75)
    ap_f.add_argument("--ix-topk-eval", type=int, default=64)
    ap_f.add_argument("--ix-energy-warmup", type=int, default=1)
    ap_f.add_argument("--ix-energy-runs", type=int, default=2)
    ap_f.set_defaults(func=cmd_phase_final)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
