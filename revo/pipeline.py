"""PhaseRunner: consolidated pipeline orchestrator for REVO Phases I-V + Ephemeral + X-XI."""
from __future__ import annotations
from dataclasses import asdict
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
from revo.ephemeral_engine import EphemeralConfig, EphemeralEngine


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


@torch.no_grad()
def _run_ephemeral_on_texts(model, tokenizer, texts, ecfg, max_len=128):
    """Run ephemeral engine on texts. Modifies model per-token but reverts fully."""
    device = next(model.parameters()).device
    engine = EphemeralEngine(model, ecfg)

    for text in texts:
        engine.reset()
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = inp.input_ids.to(device)
        slen = ids.shape[1]
        if slen < 2:
            continue

        hs = model.transformer.wte(ids)
        hs = hs + model.transformer.wpe(torch.arange(slen, device=device))
        for block in model.transformer.h:
            hs = block(hs)[0]
        hs = model.transformer.ln_f(hs)

        for pos in range(slen - 1):
            h_pos = hs[0, pos]
            logits_mod, meta = engine.step(h_pos, pos)
            if logits_mod is not None:
                logits_at = logits_mod
                target = ids[0, pos + 1]
                if logits_at.dim() == 1:
                    logits_at = logits_at.unsqueeze(0)
                logits_base = model(ids[:, :pos+1]).logits
                nll_base = float(torch.nn.functional.cross_entropy(
                    logits_base[0, -1].unsqueeze(0).float(), target.unsqueeze(0)))
                nll_mod = float(torch.nn.functional.cross_entropy(
                    logits_at.float(), target.unsqueeze(0)))
                engine.record_nll_delta(nll_mod - nll_base)


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
    def phase_ephemeral(self, model, tokenizer, texts, max_len=128):
        """Phase E: ciclo efímero completo con context encoder + gating + cache."""
        base = _em(model, tokenizer, texts, max_len)
        cfg = self.cfg.get("ephemeral", {})
        ecfg = EphemeralConfig(
            context_dim=cfg.get("context_dim", 16),
            window_size=cfg.get("window_size", 4),
            rank=cfg.get("rank", 4),
            scale_min=cfg.get("scale_min", 0.0),
            scale_max=cfg.get("scale_max", 0.8),
            use_cache=cfg.get("use_cache", True),
            cache_ttl=cfg.get("cache_ttl", 3600),
            cache_max=cfg.get("cache_max", 128),
            cache_similarity=cfg.get("cache_similarity", 0.92),
            novelty_threshold=cfg.get("novelty_threshold", 0.3),
            delta_window=cfg.get("delta_window", 3),
            delta_window_threshold=cfg.get("delta_window_threshold", 0.02),
            delta_window_scale_factor=cfg.get("delta_window_scale_factor", 0.3),
        )
        try:
            _run_ephemeral_on_texts(model, tokenizer, texts, ecfg, max_len)
            after = _em(model, tokenizer, texts, max_len)
            return _pr("phase_ephemeral", base, after,
                       extra={"ephemeral_config": asdict(ecfg)})
        except Exception as e:
            return {"name": "phase_ephemeral", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase10_primitiva_router(self, model, tokenizer, texts, max_len=128):
        base = _em(model, tokenizer, texts, max_len)
        pr = _try("revo.primitiva_router")
        if pr is None:
            return {"name": "phase10_primitiva_router", "status": "skipped"}
        try:
            cfg = self.cfg.get("phase10", {})
            pm = pr.PrimitiveModel(
                model,
                enabled=cfg.get("enabled", None),
                name_patterns=cfg.get("patterns", ["attn", "mlp", "c_fc", "c_proj"]),
                router_temperature=cfg.get("temperature", 1.0),
                router_hard=cfg.get("hard", True),
                top_k=cfg.get("top_k", None),
            )
            after = _em(pm, tokenizer, texts, max_len)
            extra = {"selector_count": len(pm._selectors),
                     "primitives": pm.enabled}
            return _pr("phase10_primitiva_router", base, after, extra=extra)
        except Exception as e:
            return {"name": "phase10_primitiva_router", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase11_generative_law(self, model, tokenizer, texts, max_len=128):
        base = _em(model, tokenizer, texts, max_len)
        gl = _try("revo.generative_law")
        if gl is None:
            return {"name": "phase11_generative_law", "status": "skipped"}
        try:
            cfg = self.cfg.get("phase11", {})
            gm = gl.GenerativeModel(
                model,
                enabled=cfg.get("enabled", None),
                name_patterns=cfg.get("patterns", ["attn", "mlp", "c_fc", "c_proj"]),
                mode=cfg.get("mode", "residual"),
                law_hidden=cfg.get("law_hidden", 64),
            )
            top_k = cfg.get("top_k", None)
            if top_k is not None:
                gm.set_top_k(top_k)
            after = _em(gm, tokenizer, texts, max_len)
            codes = gm.collect_codes()
            extra = {
                "replaced_layers": len(gm._layers),
                "mode": gm.mode,
                "codes_collected": {n: list(c.shape) for n, c in codes.items()},
                "codes_active_ratio": {
                    n: (c.sum(dim=-1).float().mean().item() / c.shape[-1])
                    for n, c in codes.items()
                } if codes else {},
            }
            return _pr("phase11_generative_law", base, after, extra=extra)
        except Exception as e:
            return {"name": "phase11_generative_law", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase12_streaming(self, model, tokenizer, texts, max_len=128):
        """Phase XII: per-layer weight streaming from disk.
        Compares full load vs streaming memory and NLL."""
        import os
        import tempfile
        from revo.streaming import shard_model, compare_memory
        base = _em(model, tokenizer, texts, max_len)
        tmp = tempfile.mkdtemp(prefix="revo_shards_")
        try:
            shard_model(model, tmp)
            report = compare_memory(model, tokenizer, tmp, texts, max_len)
            r = {
                "nll_full": report["full"]["nll"],
                "nll_stream": report["streaming"]["nll"],
                "nll_delta": report["streaming"]["nll_delta"],
                "theoretical_savings_pct": report["theory"]["savings_pct"],
                "full_weights_mb": report["theory"]["full_weights_mb"],
                "stream_peak_weights_mb": report["theory"]["stream_peak_weights_mb"],
                "time_ratio": report["streaming"]["time_ratio"],
                "rss_peak_delta_mb": report["streaming"]["rss_peak_delta_mb"],
            }
            return _pr("phase12_streaming", base, _em(model, tokenizer, texts, max_len),
                       extra=r)
        except Exception as e:
            return {"name": "phase12_streaming", "status": "error", "error": str(e)}
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def phase12b_law(self, model, tokenizer, texts, max_len=128):
        """Phase XII-b: Generative WeightLaw streaming."""
        from revo.law_streaming import (
            build_law, pretrain_law, finetune_law,
            _extract_svd_targets, law_info, _save_block_templates,
            _eval_nll_law,
        )
        from revo.streaming import _eval_nll_full, _d_model
        base = _em(model, tokenizer, texts, max_len)
        try:
            cfg = self.cfg.get("phase12b", {})
            rank = cfg.get("rank", 64)
            small_dim = cfg.get("small_dim", 32)
            hidden_dim = cfg.get("hidden_dim", 512)
            cognitive = cfg.get("cognitive", False)
            cognitive_field = cfg.get("cognitive_field", False)

            law = build_law(model, rank=rank, small_dim=small_dim,
                            hidden_dim=hidden_dim, cognitive=cognitive,
                            cognitive_field=cognitive_field)

            if not cognitive_field:
                targets = _extract_svd_targets(model, rank=rank)
                pretrain_law(law, targets, steps=cfg.get("pretrain_steps", 100),
                             lr=1e-3, verbose=False)

            if cfg.get("finetune", True) and texts and not cognitive_field:
                finetune_law(law, model, tokenizer, texts,
                             steps=cfg.get("finetune_steps", 5),
                             lr=5e-5, max_length=max_len, verbose=False)

            info = law_info(law)

            # Measure law metrics WITHOUT freeing model (saves _em at end)
            templates = _save_block_templates(model)
            from revo.streaming import _eval_nll_full, _free_all_block_weights

            nll_full = _eval_nll_full(model, tokenizer, texts, max_len)
            nll_law = _eval_nll_law(model, tokenizer, texts, law, max_len,
                                    _templates=templates)
            rss_peak = measure_memory_rss()

            extra = {
                "law_params": info["n_params"],
                "law_mb": info["size_mb"],
                "rank": info["rank"],
                "law_nll": nll_law,
                "law_nll_delta": nll_law - nll_full,
                "full_nll": nll_full,
                "law_rss_peak_mb": rss_peak / 1024 / 1024,
            }
            if hasattr(law, 'field') and law.field is not None:
                with torch.no_grad():
                    fs = law.field(torch.randn(1, 1, _d_model(model)))
                    extra["field"] = {
                        "phi": round(fs.phi.item(), 3),
                        "arousal": round(fs.arousal.item(), 3),
                        "coherence": round(fs.coherence.item(), 3),
                        "uncertainty": round(fs.uncertainty.item(), 3),
                        "active_03": len(fs.layers_to_generate(0.3)),
                        "active_05": len(fs.layers_to_generate(0.5)),
                    }
            return _pr("phase12b_law", base, _em(model, tokenizer, texts, max_len),
                       extra=extra)
        except Exception as e:
            return {"name": "phase12b_law", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase12c_cognitive_field(self, model, tokenizer, texts, max_len=128):
        """Phase XII-c: Cognitive Field + selective weight generation."""
        from revo.law_streaming import (
            build_law, law_generate, law_stream_forward,
            _save_block_templates, _eval_nll_law,
        )
        from revo.streaming import _eval_nll_full, _free_all_block_weights
        from revo.streaming import _d_model
        base = _em(model, tokenizer, texts, max_len)
        try:
            cfg = self.cfg.get("phase12c", {})
            rank = cfg.get("rank", 4)
            hidden_dim = cfg.get("hidden_dim", 64)

            law = build_law(model, rank=rank, small_dim=2,
                            hidden_dim=hidden_dim,
                            cognitive=True, cognitive_field=True)

            # Evaluate field on first text
            device = next(model.parameters()).device
            sample_ids = tokenizer(texts[0] if texts else "hello",
                                    return_tensors="pt", truncation=True,
                                    max_length=max_len).input_ids.to(device)
            from revo.streaming import _detect_arch, _extract_shared, _non_layer_pattern
            arch = _detect_arch(model)
            non_layer = _non_layer_pattern(arch)
            state = model.state_dict(keep_vars=False)
            shared = {k: v for k, v in state.items() if not non_layer.search(k)}
            wte, wpe, _, _, _ = _extract_shared(shared, arch)
            x = torch.nn.functional.embedding(sample_ids, wte.to(device))

            field_state = law.field(x)
            n_layers = law.layer_emb.shape[0]
            extra = {
                "phi": round(field_state.phi.item(), 3),
                "arousal": round(field_state.arousal.item(), 3),
                "coherence": round(field_state.coherence.item(), 3),
                "uncertainty": round(field_state.uncertainty.item(), 3),
                "scale": round(field_state.scale.item(), 3),
                "active_03": len(field_state.layers_to_generate(0.3)),
                "active_05": len(field_state.layers_to_generate(0.5)),
                "active_07": len(field_state.layers_to_generate(0.7)),
                "n_layers": n_layers,
                "law_params": sum(p.numel() for p in law.parameters()),
                "field_params": sum(p.numel() for p in law.field.parameters()),
            }

            # NLL comparison
            nll_full = _eval_nll_full(model, tokenizer, texts, max_len)
            nll_law = _eval_nll_law(model, tokenizer, texts, law, max_len)
            extra["full_nll"] = nll_full
            extra["law_nll"] = nll_law
            extra["nll_delta"] = nll_law - nll_full

            return _pr("phase12c_cognitive_field", base,
                       _em(model, tokenizer, texts, max_len), extra=extra)
        except Exception as e:
            return {"name": "phase12c_cognitive_field", "status": "error", "error": str(e)}

    @torch.no_grad()
    def phase12d_law_e2e(self, model, tokenizer, texts, max_len=128):
        """Phase XII-d: End-to-end NLL training of the WeightLaw.
        Trains law to produce weight deltas that improve NLL on wikitext.
        """
        base = _em(model, tokenizer, texts, max_len)
        try:
            from train_law_e2e import train_e2e
            cfg = self.cfg.get("phase12d", {})
            rank = cfg.get("rank", 16)
            small_dim = cfg.get("small_dim", 8)
            hidden_dim = cfg.get("hidden_dim", 256)
            steps = cfg.get("steps", 500)
            lr = cfg.get("lr", 3e-4)
            lambda_kl = cfg.get("lambda_kl", 0.0)
            max_samples = cfg.get("max_samples", 500)

            result = train_e2e(
                model_name=model.config._name_or_path if hasattr(model.config, '_name_or_path') else str(type(model).__name__),
                rank=rank, small_dim=small_dim, hidden_dim=hidden_dim,
                lr=lr, steps=steps, batch_size=cfg.get("batch_size", 2),
                max_samples=max_samples, lambda_kl=lambda_kl,
            )
            return _pr("phase12d_law_e2e", base,
                       _em(model, tokenizer, texts, max_len),
                       extra={"status": "completed",
                              "law_params": result.get("law_params", 0),
                              "best_val_nll": result.get("best_val_nll", 0),
                              "nll_delta": result.get("nll_delta", 0)})
        except Exception as e:
            return {"name": "phase12d_law_e2e", "status": "error", "error": str(e)}

    @torch.no_grad()
    def run_all(self, model, tokenizer, texts_general, texts_ood=None, max_len=128):
        report = {"baseline": _em(model, tokenizer, texts_general, max_len)}
        phases = [
            ("phase1", self.phase1_metric_field),
            ("phase2", self.phase2_holography),
            ("phase3", self.phase3_spectral),
            ("phase4", self.phase4_fractal),
            ("phase5", self.phase5_energy),
            ("phase_ephemeral", self.phase_ephemeral),
            ("phase10", self.phase10_primitiva_router),
            ("phase11", self.phase11_generative_law),
            ("phase12", self.phase12_streaming),
            ("phase12b", self.phase12b_law),
            ("phase12c", self.phase12c_cognitive_field),
            ("phase12d", self.phase12d_law_e2e),
        ]
        for phase, fn in phases:
            report[phase] = fn(model, tokenizer, texts_general, max_len)
        if texts_ood:
            report["ood_baseline"] = _em(model, tokenizer, texts_ood, max_len)
        completed = [k for k, v in report.items() if isinstance(v, dict) and v.get("status") == "completed"]
        skipped = [k for k, v in report.items() if isinstance(v, dict) and v.get("status") == "skipped"]
        report["summary"] = {"phases_completed": completed, "phases_skipped": skipped,
                             "total_phases": len(phases), "completed_count": len(completed)}
        return report


def run_pipeline_cli() -> None:
    ap = argparse.ArgumentParser(description="REVO Pipeline CLI (Phases I-V + X-XI)")
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
    for key in ["phase1", "phase2", "phase3", "phase4", "phase5", "phase_ephemeral",
                 "phase10", "phase11", "phase12", "phase12b", "phase12c", "phase12d"]:
        p = report.get(key, {})
        nd = f"{p.get('nll_delta', 0):+.6e}" if p.get("nll_delta") is not None else "—"
        tr = f"{p.get('time_ratio', 1):.4f}×" if p.get("time_ratio") is not None else "—"
        pd = f"{p.get('params_delta', 0):+d}" if p.get("params_delta") is not None else "—"
        print(f"{key:<20} {p.get('status','?'):<12} {nd:<14} {tr:<12} {pd:<12}")
    print(f"{'─'*60}\nBaseline NLL: {report.get('baseline',{}).get('nll','—'):.6f}")


if __name__ == "__main__":
    run_pipeline_cli()
