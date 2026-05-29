"""PhaseRunner: consolidated pipeline orchestrator for REVO Phases I-V."""
from __future__ import annotations
from revo._logging import get_logger

import argparse
import importlib
import time
from typing import Any, Dict, List, Optional

import torch

from revo._utils import (
    count_parameters, evaluate_nll, gen_texts,
    load_model_tokenizer, measure_memory_rss, save_results_json,
)


def _em(model, tok, texts, max_len):
    r0 = measure_memory_rss()
    t0 = time.perf_counter()
    nll = evaluate_nll(model, tok, texts, max_length=max_len)
    dt = time.perf_counter() - t0
    r1 = measure_memory_rss()
    return {"nll": nll, "eval_time_s": dt, "params": count_parameters(model), "rss_bytes": r1, "rss_delta_bytes": int(r1 - r0)}


def _pr(name, before, after, status="completed", extra=None):
    nd = after["nll"] - before["nll"]
    tr = after["eval_time_s"] / before["eval_time_s"] if before["eval_time_s"] > 0 else 1.0
    pd = after["params"] - before["params"]
    r = {"name": name, "nll": after["nll"], "nll_delta": nd, "eval_time_s": after["eval_time_s"], "time_ratio": tr, "params": after["params"], "params_delta": pd, "rss_bytes": after["rss_bytes"], "status": status}
    if extra:
        r["extra"] = extra
    return r


def _try(mod_name):
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        return None


class PhaseRunner:
    """Orchestrates REVO Phases I-V in sequence with per-phase metrics."""

    def __init__(self, phase_config: Optional[Dict[str, Any]] = None):
        self.cfg = phase_config or {}

    @torch.no_grad()
    def phase1_metric_field(self, model, tokenizer=None, texts=None, max_len=128, embeddings=None):
        base = _em(model, tokenizer, texts or [], max_len) if texts else {"nll": 0, "eval_time_s": 1, "params": count_parameters(model), "rss_bytes": measure_memory_rss()}
        ep = _try("revo.metric_field_pinn")
        if ep is not None and hasattr(ep, "compute_curvature"):
            try:
                ep.compute_curvature(model, embeddings)
                return _pr("phase1_metric_field", base, _em(model, tokenizer, texts or [], max_len))
            except Exception:
                get_logger().warning("except Exception:")
        lp, lr = _try("revo.layer_profile"), _try("revo.lowrank")
        if lp is None or lr is None:
            return {"name": "phase1_metric_field", "status": "skipped"}
        try:
            c = self.cfg.get("phase1", {})
            prof = lp.profile_model_2d(model, name_patterns=c.get("patterns", ["attn", "mlp", "c_fc", "c_proj"]), max_rank=c.get("max_rank"))
            ranks = lp.allocate_ranks_energy_with_caps(prof, energy_keep=c.get("energy_keep", 0.92), max_rank=c.get("max_rank"), max_rank_frac=c.get("max_rank_frac", 0.25))
            lr.replace_2d_modules_with_lowrank(model, ranks, calibrate=bool(c.get("calibrate", False)), calibrate_samples=256, seed=c.get("seed", 0))
            return _pr("phase1_metric_field", base, _em(model, tokenizer, texts or [], max_len))
        except Exception as e:
            return {"name": "phase1_metric_field", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase2_holography(self, model, tokenizer, texts, max_len=128):
        base = _em(model, tokenizer, texts, max_len)
        hora, holo = _try("revo.hora"), _try("revo.holography")
        if hora is None and holo is None:
            return {"name": "phase2_holography", "status": "skipped"}
        try:
            cfg, p = self.cfg.get("phase2", {}), ["attn", "mlp", "c_fc", "c_proj"]
            if hora is not None and cfg.get("use_hora", True):
                hora.replace_with_hora(model, rank=cfg.get("rank", 4), alpha=cfg.get("alpha", 8.0), c=cfg.get("c_val", 0.05), name_patterns=p, skip_lm_head=True)
            if holo is not None and cfg.get("use_holo", True):
                holo.replace_with_holography(model, boundary_dim=cfg.get("boundary_dim", 16), alpha=cfg.get("holo_alpha", 0.2), name_patterns=p, skip_lm_head=True)
                holo.calibrate_holo_pinn(model, tokenizer, texts=texts, steps=cfg.get("pinn_steps", 10), lr=1e-2, lambda_phys=cfg.get("pinn_lambda", 1.0), max_length=max_len)
            return _pr("phase2_holography", base, _em(model, tokenizer, texts, max_len))
        except Exception as e:
            return {"name": "phase2_holography", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase3_spectral(self, model, tokenizer, texts, max_len=128):
        base = _em(model, tokenizer, texts, max_len)
        sp, pb, rv, wd, fk = [_try(n) for n in ["revo.spectral", "revo.phase_bus", "revo.reversible", "revo.wdm", "revo.fft_kernel"]]
        if all(m is None for m in [sp, pb, rv, wd, fk]):
            return {"name": "phase3_spectral", "status": "skipped"}
        try:
            cfg, p = self.cfg.get("phase3", {}), ["attn", "mlp", "c_fc", "c_proj"]
            if sp is not None:
                sp.prune_model_spectral(model, energy_keep=cfg.get("energy_keep", 0.90), name_patterns=p, skip_lm_head=True)
            if pb is not None:
                pb.replace_with_phase_bus(model, name_patterns=p, skip_lm_head=True)
                pb.calibrate_phase_bus(model, tokenizer, texts=texts, steps=cfg.get("phase_steps", 10), lr=5e-2, lambda_phys=cfg.get("phase_lambda", 1.0), max_length=max_len)
            if rv is not None:
                rv.replace_with_reversible(model, rank=cfg.get("rev_rank", 2), name_patterns=p, skip_lm_head=True)
                rv.calibrate_reversible(model, tokenizer, texts=texts, steps=cfg.get("rev_steps", 10), lr=5e-2, lambda_phys=cfg.get("rev_lambda", 1.0), max_length=max_len)
            if wd is not None:
                wd.replace_with_wdm(model, bands=cfg.get("bands", 2), name_patterns=p, skip_lm_head=True)
            if fk is not None:
                fk.replace_with_circulant(model, name_patterns=p, skip_lm_head=True)
            return _pr("phase3_spectral", base, _em(model, tokenizer, texts, max_len))
        except Exception as e:
            return {"name": "phase3_spectral", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase4_fractal(self, model, tokenizer, texts, max_len=128):
        base = _em(model, tokenizer, texts, max_len)
        fr, ep, rx = _try("revo.fractal"), _try("revo.ephemeral"), _try("revo.radix")
        if all(m is None for m in [fr, ep, rx]):
            return {"name": "phase4_fractal", "status": "skipped"}
        extra = {}
        try:
            cfg = self.cfg.get("phase4", {})
            if fr is not None:
                fr.replace_with_fractal(model, depth=cfg.get("depth", 2), alpha=cfg.get("fractal_alpha", 0.5), name_patterns=cfg.get("fractal_patterns", ["attn", "mlp", "c_proj"]), skip_lm_head=True)
            if ep is not None:
                ep.replace_with_ephemeral(model, name_patterns=cfg.get("patterns", ["attn", "mlp", "c_fc", "c_proj"]), skip_lm_head=True)
                ep.calibrate_ephemeral(model, tokenizer, texts=texts, steps=cfg.get("epi_steps", 10), lr=5e-2, lambda_phys=cfg.get("epi_lambda", 1.0), max_length=max_len)
            if rx is not None:
                extra["radix_eval"] = rx.radix_eval_nll(model, tokenizer, texts, max_length=max_len)
            return _pr("phase4_fractal", base, _em(model, tokenizer, texts, max_len), extra=extra or None)
        except Exception as e:
            return {"name": "phase4_fractal", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase5_energy(self, model, tokenizer, texts, max_len=128):
        base = _em(model, tokenizer, texts, max_len)
        eg, ml = _try("revo.energy"), _try("revo.mlir_kernels")
        if eg is None:
            return {"name": "phase5_energy", "status": "skipped"}
        extra = {}
        try:
            c = self.cfg.get("phase5", {})
            if ml is not None and c.get("compile", False):
                ml.compile_model_guarded(model)
            er = eg.measure_energy(model, tokenizer, texts, max_length=max_len)
            extra["energy"] = {"total_flops": er.total_flops, "dyn_energy_j": er.dyn_energy_j, "landauer_lower_j": er.landauer_lower_j, "latency_s": er.latency_s, "tokens": er.tokens}
            return _pr("phase5_energy", base, _em(model, tokenizer, texts, max_len), extra=extra or None)
        except Exception as e:
            return {"name": "phase5_energy", "status": "error", "error": str(e)}

    @torch.no_grad()
    def run_all(self, model, tokenizer, texts_general, texts_ood=None, max_len=128):
        report = {"baseline": _em(model, tokenizer, texts_general, max_len)}
        for phase, fn in [("phase1", self.phase1_metric_field), ("phase2", self.phase2_holography), ("phase3", self.phase3_spectral), ("phase4", self.phase4_fractal), ("phase5", self.phase5_energy)]:
            report[phase] = fn(model, tokenizer, texts_general, max_len)
        if texts_ood:
            report["ood_baseline"] = _em(model, tokenizer, texts_ood, max_len)
        completed = [k for k, v in report.items() if isinstance(v, dict) and v.get("status") == "completed"]
        skipped = [k for k, v in report.items() if isinstance(v, dict) and v.get("status") == "skipped"]
        report["summary"] = {"phases_completed": completed, "phases_skipped": skipped, "total_phases": 5, "completed_count": len(completed)}
        return report


def run_pipeline_cli() -> None:
    ap = argparse.ArgumentParser(description="REVO Pipeline CLI (Phases I-V)")
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--prompts", type=int, default=50)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--results-json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model_tokenizer(args.model)
    report = PhaseRunner({"phase1": {"seed": args.seed}}).run_all(model, tokenizer, gen_texts(args.prompts), max_len=args.max_length)
    path = save_results_json(report, default_dir="quality", prefix="pipeline", name=args.results_json or "")
    print(f"Pipeline results saved to {path}")
    print(f"\n{'─'*60}\n{'Phase':<20} {'Status':<12} {'NLL Δ':<14} {'Time ratio':<12} {'Params Δ':<12}\n{'─'*60}")
    for key in ["phase1", "phase2", "phase3", "phase4", "phase5"]:
        p = report.get(key, {})
        nd = f"{p.get('nll_delta', 0):+.6e}" if p.get("nll_delta") is not None else "—"
        tr = f"{p.get('time_ratio', 1):.4f}×" if p.get("time_ratio") is not None else "—"
        pd = f"{p.get('params_delta', 0):+d}" if p.get("params_delta") is not None else "—"
        print(f"{key:<20} {p.get('status','?'):<12} {nd:<14} {tr:<12} {pd:<12}")
    print(f"{'─'*60}\nBaseline NLL: {report.get('baseline',{}).get('nll','—'):.6f}")


if __name__ == "__main__":
    run_pipeline_cli()
