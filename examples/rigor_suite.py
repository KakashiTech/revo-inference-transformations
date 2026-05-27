from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Phase I
from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank
# Phase II
try:
    from revo.hora import replace_with_hora  # type: ignore
except Exception:  # pragma: no cover
    replace_with_hora = None
try:
    from revo.holography import replace_with_holography, calibrate_holo_pinn  # type: ignore
except Exception:  # pragma: no cover
    replace_with_holography = None
    calibrate_holo_pinn = None
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


def _measure_rss_bytes() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


def _param_count(model: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _gen_texts_general(n: int) -> List[str]:
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
    return [base[i % len(base)] + f" [G#{i}]" for i in range(n)]


def _gen_texts_ood(n: int) -> List[str]:
    base = [
        "Summarize the role of spectral sparsity in compression.",
        "Write a short Python function for radix prefix search.",
        "Explain energy vs Landauer limit in simple terms.",
        "What is reversible computing and why does it matter?",
        "Give an example of wave-division multiplexing in optics.",
    ]
    return [base[i % len(base)] + f" [O#{i}]" for i in range(n)]


def _gen_texts_stress(n: int, repeat: int = 64) -> List[str]:
    long_seed = (
        "Este es un texto largo diseñado para estresar el contexto y medir latencia. "
        "La prueba incluye repetición para inducir cargas sostenidas. "
    )
    return [(long_seed * repeat) + f" [S#{i}]" for i in range(n)]


def _eval_losses(model, tok, texts: List[str], max_length: int = 256) -> Tuple[List[float], List[int], float, float]:
    model.eval()
    losses: List[float] = []  # per-sample mean NLL (per token)
    tok_counts: List[int] = []
    rss0 = _measure_rss_bytes()
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc.input_ids
            attn = enc.attention_mask
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            loss = float(out.loss.item())  # mean over tokens in this sample
            n_tok = int(input_ids.numel())
            losses.append(loss)
            tok_counts.append(n_tok)
    dt = time.perf_counter() - t0
    rss1 = _measure_rss_bytes()
    return losses, tok_counts, dt, float(rss1 - rss0)


def _cohens_d(x: List[float], y: List[float]) -> float:
    if not x or not y:
        return 0.0
    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    diff = x_arr - y_arr
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def _bootstrap_ci(deltas: List[float], iters: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = np.random.default_rng(123)
    arr = np.array(deltas, dtype=np.float64)
    boots = []
    n = len(arr)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(arr[idx])))
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _sign_test_pvalue(deltas: List[float]) -> float:
    # Two-sided sign test using normal approximation
    if not deltas:
        return 1.0
    n = len(deltas)
    k = sum(1 for d in deltas if d > 0)
    # z = (k - n/2)/sqrt(n/4)
    denom = math.sqrt(max(1e-12, n / 4.0))
    z = (k - n / 2.0) / denom
    # two-sided p-value from normal
    return float(2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0)))


@dataclass
class ExperimentConfig:
    name: str
    p1: bool = False
    p1_energy_keep: float = 0.92
    p1_max_rank: int | None = None
    p1_max_rank_frac: float = 0.25
    p1_calibrate: bool = True

    p2_hora: bool = False
    p2_holo: bool = False
    p2_rank: int = 4
    p2_alpha: float = 8.0
    p2_c: float = 0.05
    p2_boundary_dim: int = 16
    p2_holo_alpha: float = 0.2
    p2_pinn_steps: int = 10
    p2_pinn_lambda: float = 1.0

    p3: bool = False
    p3_energy_keep: float = 0.90
    p3_phase_steps: int = 10
    p3_phase_lambda: float = 1.0
    p3_rev_rank: int = 2
    p3_rev_steps: int = 10
    p3_rev_lambda: float = 1.0
    p3_bands: int = 2

    p4: bool = False
    p4_depth: int = 2
    p4_alpha: float = 0.5
    p4_epi_steps: int = 10
    p4_epi_lambda: float = 1.0

    measure_latency_energy: bool = True


def _apply_transforms(model, tok, texts, args, cfg: ExperimentConfig) -> None:
    name_patterns = ["attn", "mlp", "c_fc", "c_proj"]

    if cfg.p1:
        prof = profile_model_2d(model, name_patterns=name_patterns, max_rank=cfg.p1_max_rank)
        ranks = allocate_ranks_energy_with_caps(
            prof,
            energy_keep=cfg.p1_energy_keep,
            max_rank=cfg.p1_max_rank,
            max_rank_frac=cfg.p1_max_rank_frac,
        )
        replace_2d_modules_with_lowrank(model, ranks, calibrate=bool(cfg.p1_calibrate), calibrate_samples=256, seed=0)

    if cfg.p2_hora and replace_with_hora is not None:
        try:
            replace_with_hora(model, rank=cfg.p2_rank, alpha=cfg.p2_alpha, c=cfg.p2_c, name_patterns=name_patterns, skip_lm_head=True)
        except Exception:
            pass
    if cfg.p2_holo and replace_with_holography is not None and calibrate_holo_pinn is not None:
        try:
            replace_with_holography(model, boundary_dim=cfg.p2_boundary_dim, alpha=cfg.p2_holo_alpha, name_patterns=name_patterns, skip_lm_head=True)
            calibrate_holo_pinn(model, tok, texts=texts, steps=cfg.p2_pinn_steps, lr=1e-2, lambda_phys=cfg.p2_pinn_lambda, max_length=args.max_length)
        except Exception:
            pass

    if cfg.p3:
        prune_model_spectral(model, energy_keep=cfg.p3_energy_keep, name_patterns=name_patterns, skip_lm_head=True)
        replace_with_phase_bus(model, name_patterns=name_patterns, skip_lm_head=True)
        calibrate_phase_bus(model, tok, texts=texts, steps=cfg.p3_phase_steps, lr=5e-2, lambda_phys=cfg.p3_phase_lambda, max_length=args.max_length)
        replace_with_reversible(model, rank=cfg.p3_rev_rank, name_patterns=name_patterns, skip_lm_head=True)
        calibrate_reversible(model, tok, texts=texts, steps=cfg.p3_rev_steps, lr=5e-2, lambda_phys=cfg.p3_rev_lambda, max_length=args.max_length)
        replace_with_wdm(model, bands=cfg.p3_bands, name_patterns=name_patterns, skip_lm_head=True)
        replace_with_circulant(model, name_patterns=name_patterns, skip_lm_head=True)

    if cfg.p4:
        replace_with_fractal(model, depth=cfg.p4_depth, alpha=cfg.p4_alpha, name_patterns=["attn", "mlp", "c_proj"], skip_lm_head=True)
        replace_with_ephemeral(model, name_patterns=name_patterns, skip_lm_head=True)
        calibrate_ephemeral(model, tok, texts=texts, steps=cfg.p4_epi_steps, lr=5e-2, lambda_phys=cfg.p4_epi_lambda, max_length=args.max_length)


@dataclass
class RunResult:
    seed: int
    exp_name: str
    set_name: str
    n_params: int
    rss_delta_bytes: float
    eval_time_s: float
    mean_nll: float
    per_sample_nll: List[float]
    tokens: List[int]
    latency: Dict[str, Any] | None
    energy: Dict[str, Any] | None


@dataclass
class ExperimentSummary:
    name: str
    set_name: str
    seeds: List[int]
    mean_nll: float
    std_nll: float
    mean_time_s: float
    params_mean: float
    rss_delta_mean: float
    effect_size_vs_baseline: float
    ci95_delta_vs_baseline: Tuple[float, float]
    sign_test_p_vs_baseline: float


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO Rigor Suite: ablations, stress, stats")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--include-stress", action="store_true")
    ap.add_argument("--include-ood", action="store_true")
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Define experiment grid
    exps: List[ExperimentConfig] = [
        ExperimentConfig(name="baseline", p1=False, p2_hora=False, p2_holo=False, p3=False, p4=False, measure_latency_energy=True),
        ExperimentConfig(name="P1_mod", p1=True, p1_energy_keep=0.95, p1_max_rank=None, p1_max_rank_frac=0.20),
        ExperimentConfig(name="P1_agg", p1=True, p1_energy_keep=0.90, p1_max_rank=None, p1_max_rank_frac=0.15),
        ExperimentConfig(name="P1+P2", p1=True, p1_energy_keep=0.92, p2_hora=True, p2_holo=True),
        ExperimentConfig(name="P1+P2+P3", p1=True, p2_hora=True, p2_holo=True, p3=True, p3_energy_keep=0.90),
        ExperimentConfig(name="P1+P2+P3+P4", p1=True, p2_hora=True, p2_holo=True, p3=True, p4=True),
        ExperimentConfig(name="All_minus_P2", p1=True, p3=True, p4=True),
        ExperimentConfig(name="All_minus_P3", p1=True, p2_hora=True, p2_holo=True, p4=True),
        ExperimentConfig(name="All_minus_P4", p1=True, p2_hora=True, p2_holo=True, p3=True),
    ]

    # Text sets to evaluate
    sets: List[Tuple[str, List[str]]] = [("general", _gen_texts_general(args.prompts))]
    if args.include_ood:
        sets.append(("ood", _gen_texts_ood(args.prompts)))
    if args.include_stress:
        sets.append(("stress", _gen_texts_stress(max(10, args.prompts // 5), repeat=64)))

    all_runs: List[RunResult] = []

    def _env_info() -> Dict[str, Any]:
        return {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "hostname": platform.node(),
            "os": platform.platform(),
        }

    for set_name, texts in sets:
        for exp in exps:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                tok = AutoTokenizer.from_pretrained(args.model)
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                model = AutoModelForCausalLM.from_pretrained(args.model)
                model.eval()

                # Baseline measure before transforms for RSS/params reference (not stored)
                # Apply transforms for this experiment (no-op for baseline)
                try:
                    _apply_transforms(model, tok, texts, args, exp)
                except Exception:
                    # Defensive: still evaluate if some transform failed
                    pass

                losses, tokens, dt, rss_delta = _eval_losses(model, tok, texts, max_length=args.max_length)
                latency = None
                energy = None
                if exp.measure_latency_energy:
                    try:
                        lat = measure_latency_distribution(model, tok, texts, max_length=args.max_length, warmup=1, runs=3)
                        latency = lat
                    except Exception:
                        latency = None
                    try:
                        en = measure_energy(model, tok, texts, max_length=args.max_length)
                        energy = {
                            "total_flops": en.total_flops,
                            "dyn_energy_j": en.dyn_energy_j,
                            "landauer_lower_j": en.landauer_lower_j,
                            "latency_s": en.latency_s,
                            "tokens": en.tokens,
                        }
                    except Exception:
                        energy = None

                all_runs.append(
                    RunResult(
                        seed=seed,
                        exp_name=exp.name,
                        set_name=set_name,
                        n_params=_param_count(model),
                        rss_delta_bytes=float(rss_delta),
                        eval_time_s=float(dt),
                        mean_nll=float(np.mean(losses) if losses else 0.0),
                        per_sample_nll=[float(x) for x in losses],
                        tokens=[int(t) for t in tokens],
                        latency=latency,
                        energy=energy,
                    )
                )

    # Aggregate and compute stats vs baseline per set and experiment
    summaries: List[ExperimentSummary] = []
    by_key: Dict[Tuple[str, str], List[RunResult]] = {}
    for r in all_runs:
        by_key.setdefault((r.exp_name, r.set_name), []).append(r)

    # Gather per-prompt deltas vs baseline (paired across seeds by index)
    baselines: Dict[str, List[RunResult]] = {set_name: by_key.get(("baseline", set_name), []) for set_name, _ in sets}

    for (exp_name, set_name), runs in by_key.items():
        if exp_name == "baseline":
            # Summarize baseline itself
            mean_nll = float(np.mean([r.mean_nll for r in runs])) if runs else 0.0
            std_nll = float(np.std([r.mean_nll for r in runs], ddof=1)) if len(runs) > 1 else 0.0
            mean_time = float(np.mean([r.eval_time_s for r in runs])) if runs else 0.0
            mean_params = float(np.mean([r.n_params for r in runs])) if runs else 0.0
            mean_rss = float(np.mean([r.rss_delta_bytes for r in runs])) if runs else 0.0
            summaries.append(
                ExperimentSummary(
                    name=exp_name,
                    set_name=set_name,
                    seeds=[r.seed for r in runs],
                    mean_nll=mean_nll,
                    std_nll=std_nll,
                    mean_time_s=mean_time,
                    params_mean=mean_params,
                    rss_delta_mean=mean_rss,
                    effect_size_vs_baseline=0.0,
                    ci95_delta_vs_baseline=(0.0, 0.0),
                    sign_test_p_vs_baseline=1.0,
                )
            )
            continue

        base_runs = baselines.get(set_name, [])
        # Build deltas pairing by seed and sample index
        deltas: List[float] = []
        base_map: Dict[int, RunResult] = {r.seed: r for r in base_runs}
        for r in runs:
            b = base_map.get(r.seed)
            if not b:
                continue
            m = min(len(r.per_sample_nll), len(b.per_sample_nll))
            if m == 0:
                continue
            for i in range(m):
                deltas.append(float(b.per_sample_nll[i] - r.per_sample_nll[i]))  # positive means improvement

        mean_nll = float(np.mean([rr.mean_nll for rr in runs])) if runs else 0.0
        std_nll = float(np.std([rr.mean_nll for rr in runs], ddof=1)) if len(runs) > 1 else 0.0
        mean_time = float(np.mean([rr.eval_time_s for rr in runs])) if runs else 0.0
        mean_params = float(np.mean([rr.n_params for rr in runs])) if runs else 0.0
        mean_rss = float(np.mean([rr.rss_delta_bytes for rr in runs])) if runs else 0.0

        eff_size = _cohens_d([x for r0 in base_runs for x in r0.per_sample_nll], [x for r1 in runs for x in r1.per_sample_nll])
        ci_lo, ci_hi = _bootstrap_ci(deltas, iters=1000, alpha=0.05)
        p_sign = _sign_test_pvalue(deltas)

        summaries.append(
            ExperimentSummary(
                name=exp_name,
                set_name=set_name,
                seeds=[r.seed for r in runs],
                mean_nll=mean_nll,
                std_nll=std_nll,
                mean_time_s=mean_time,
                params_mean=mean_params,
                rss_delta_mean=mean_rss,
                effect_size_vs_baseline=float(eff_size),
                ci95_delta_vs_baseline=(float(ci_lo), float(ci_hi)),
                sign_test_p_vs_baseline=float(p_sign),
            )
        )

    ts = int(time.time())
    out_path = args.results_json or os.path.join("quality", "rigor", f"rigor_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "config": {
            "model": args.model,
            "seeds": seeds,
            "prompts": args.prompts,
            "max_length": args.max_length,
            "include_stress": bool(args.include_stress),
            "include_ood": bool(args.include_ood),
            "env": _env_info(),
        },
        "experiments": [asdict(e) for e in exps],
        "runs": [asdict(r) for r in all_runs],
        "summaries": [asdict(s) for s in summaries],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({"results_path": out_path, "n_runs": len(all_runs)}, indent=2))


if __name__ == "__main__":
    main()
