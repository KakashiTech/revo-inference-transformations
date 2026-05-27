from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import gc
import types
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import count_parameters, free_memory_trim, measure_memory_rss

from revo.energy import measure_energy, DEFAULT_T_K
from revo.mei_sync import measure_latency_distribution
from revo.mlir_kernels import compile_model_guarded
from revo.reversible import replace_with_reversible, calibrate_reversible, calibrate_reversible_mdl, DecomputeManager
from revo.pdm import pdm_eval_lm_head
from revo.functor_mc import verify_functor_mapping
from revo.cauchynet import replace_mlp_with_cauchy
from revo.morse import morse_skeletonize
from revo.eqprop import equilibrium_propagation_tune
from revo.holomorphic import replace_mlp_with_holomorphic, calibrate_holomorphic
from revo.phase4 import (
    solomonoff_mixed_nll,
    compositional_consistency,
    hyperbolic_profile,
    mdl_surrogate_nll,
)
from revo.phase3 import (
    OscillatoryHooks,
    interference_metrics,
    oscillatory_bptt_tune,
)


def _texts_default() -> List[str]:
    return [
        "Resume en dos líneas el principio de Landauer aplicado a IA.",
        "Explain how synchronizing Mass-Energy-Information can stabilize latency.",
        "¿Qué beneficio tiene compilar kernels (torch.compile) para inferencia?",
    ]


def _eval_nll(model, tok, texts: List[str], max_length: int = 128, trim_each_prompt: bool = False) -> Tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    # inference_mode reduces allocator churn vs no_grad on CPU
    with torch.inference_mode():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc.input_ids
            attn = enc.attention_mask
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            loss = float(out.loss.item())
            n_tok = int(input_ids.numel())
            total_loss += loss * n_tok
            total_tokens += n_tok
            if trim_each_prompt:
                try:
                    gc.collect()
                except Exception:
                    pass
                try:
                    if DecomputeManager.enabled:
                        DecomputeManager.maybe_trim()
                except Exception:
                    pass
                try:
                    free_memory_trim()
                except Exception:
                    pass
    mean_nll = float(total_loss / max(1, total_tokens))
    return mean_nll, total_tokens


def _eval_nll_osc_interf(
    model,
    tok,
    texts: List[str],
    manager: OscillatoryHooks,
    passes: int = 2,
    dphi: float = 0.25,
    max_length: int = 128,
    agg_mode: str = "logits",
    trim_each_prompt: bool = False,
) -> Tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        base_ph = manager.phases()
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc.input_ids
            attn = enc.attention_mask
            logits_acc = None
            p_sum = None  # for 'mixture' aggregator
            for i in range(max(1, int(passes))):
                # restore to base phases
                curr = manager.phases()
                manager.add_phase_vector({k: float(base_ph[k] - curr.get(k, base_ph[k])) for k in base_ph})
                # apply +/- offset
                offset = float(dphi if (i % 2 == 0) else -dphi)
                manager.add_phase_vector({k: offset for k in base_ph})
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                logits = out.logits  # [B,T,V]
                if agg_mode == "mixture":
                    # accumulate probabilities of true labels across passes (mixture-of-softmax)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    logZ = torch.logsumexp(shift_logits, dim=-1)  # [B,T-1]
                    score_y = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B,T-1]
                    p_y = torch.exp(score_y - logZ)  # [B,T-1]
                    p_sum = p_y if p_sum is None else (p_sum + p_y)
                else:
                    # default: average logits
                    logits_acc = logits if logits_acc is None else (logits_acc + logits)
                del logits, out
            if agg_mode == "mixture":
                p_avg = p_sum / float(max(1, int(passes)))
                # avoid log(0)
                p_avg = torch.clamp(p_avg, min=1e-12)
                loss = -torch.log(p_avg).sum()
                total_loss += float(loss.item())
                total_tokens += int(p_avg.numel())
            else:
                logits_avg = logits_acc / float(max(1, int(passes)))
                shift_logits = logits_avg[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                total_loss += float(loss.item())
                total_tokens += int(shift_labels.numel())
            if trim_each_prompt:
                try:
                    gc.collect()
                except Exception:
                    pass
                try:
                    if DecomputeManager.enabled:
                        DecomputeManager.maybe_trim()
                except Exception:
                    pass
                try:
                    free_memory_trim()
                except Exception:
                    pass
    mean_nll = float(total_loss / max(1, total_tokens))
    return mean_nll, total_tokens


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO Phase V Pipeline: Reversible, Adiabatic, PDM + Energy/Latency/Compile")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("REVO_SEED", "0") or 0))
    # Reversible attention/MLP proxy
    ap.add_argument("--use-reversible", action="store_true")
    ap.add_argument("--rev-rank", type=int, default=2)
    ap.add_argument("--rev-steps", type=int, default=0)
    ap.add_argument("--rev-lambda", type=float, default=1.0)
    ap.add_argument("--enable-decompute", action="store_true")
    ap.add_argument("--decompute-interval", type=int, default=0)
    # PDM proxy
    ap.add_argument("--use-pdm", action="store_true")
    ap.add_argument("--pdm-bits", type=int, default=32)
    ap.add_argument("--pdm-topk", type=int, default=256)
    # Adiabatic energy recovery proxy
    ap.add_argument("--adiabatic-eta", type=float, default=0.0)
    # Functor mapping (micro-controller) proxy
    ap.add_argument("--use-functor-mc", action="store_true")
    ap.add_argument("--functor-max-rows", type=int, default=64)
    ap.add_argument("--functor-max-samples", type=int, default=16)
    # Phase 1 (CauchyNet EinFields)
    ap.add_argument("--use-cauchy", action="store_true")
    ap.add_argument("--cauchy-rank", type=int, default=4)
    ap.add_argument("--cauchy-patterns", type=str, default=None)
    # Phase 2 (Morse Cancellation)
    ap.add_argument("--use-morse", action="store_true")
    ap.add_argument("--morse-keep-frac", type=float, default=0.95)
    ap.add_argument("--morse-patterns", type=str, default=None)
    # Phase 4 metrics
    ap.add_argument("--use-solomonoff", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--use-topos-proxy", action="store_true")
    ap.add_argument("--use-hyper-profile", action="store_true")
    ap.add_argument("--use-mdl", action="store_true")
    ap.add_argument("--mdl-lambda", type=float, default=0.01)
    ap.add_argument("--mdl-in-loss", action="store_true")
    ap.add_argument("--mdl-in-loss-steps", type=int, default=0)
    ap.add_argument("--mdl-in-loss-lambda", type=float, default=0.01)
    # Phase 3 ONN and BPTT/EP
    ap.add_argument("--use-osc", action="store_true")
    ap.add_argument("--osc-alpha", type=float, default=0.05)
    ap.add_argument("--osc-freq", type=float, default=0.25)
    ap.add_argument("--osc-kappa", type=float, default=0.0)
    ap.add_argument("--osc-steps", type=int, default=0)
    ap.add_argument("--interf-dphi", type=float, default=0.25)
    ap.add_argument("--bptt-steps", type=int, default=0)
    ap.add_argument("--bptt-lr", type=float, default=5e-2)
    ap.add_argument("--use-ep", action="store_true")
    ap.add_argument("--ep-steps", type=int, default=0)
    ap.add_argument("--ep-lr", type=float, default=5e-2)
    ap.add_argument("--ep-beta", type=float, default=0.5)
    ap.add_argument("--use-compile", action="store_true")
    ap.add_argument("--compile-backend", type=str, default="inductor")
    ap.add_argument("--energy-T-K", type=float, default=DEFAULT_T_K)
    ap.add_argument("--latency-warmup", type=int, default=2)
    ap.add_argument("--latency-runs", type=int, default=5)
    ap.add_argument("--patterns", type=str, default="attn,mlp,c_proj")
    ap.add_argument("--skip-lm-head", action="store_true")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--results-json", default=None)
    ap.add_argument("--prompts", nargs="+", default=None)
    ap.add_argument("--prompts-file", type=str, default=None)
    # Holomorphic substitution (Cauchy integral-inspired)
    ap.add_argument("--use-holomorphic", action="store_true")
    ap.add_argument("--holo-rank", type=int, default=8)
    ap.add_argument("--holo-patterns", type=str, default=None)
    ap.add_argument("--holo-calib-steps", type=int, default=0)
    ap.add_argument("--holo-mdl-lambda", type=float, default=0.0)
    # Oscillatory inference (interference averaging)
    ap.add_argument("--osc-infer", action="store_true")
    ap.add_argument("--osc-passes", type=int, default=2)
    ap.add_argument("--osc-dphi", type=float, default=0.25)
    ap.add_argument("--osc-agg", type=str, default="logits")
    # System-level controls for 100-prompt stability
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-energy", action="store_true")
    ap.add_argument("--trim-each-prompt", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    if args.enable_decompute:
        try:
            DecomputeManager.enable(trim_interval=int(args.decompute_interval))
        except Exception:
            pass

    # Select texts
    if args.prompts_file is not None:
        try:
            with open(args.prompts_file, "r", encoding="utf-8") as f:
                texts = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            texts = args.prompts if args.prompts is not None else _texts_default()
    else:
        texts = args.prompts if args.prompts is not None else _texts_default()

    # Baseline metrics
    t0 = time.perf_counter()
    base_nll, _ = _eval_nll(model, tok, texts, max_length=args.max_length, trim_each_prompt=bool(args.trim_each_prompt))
    base_time = time.perf_counter() - t0
    base_params = count_parameters(model)
    try:
        gc.collect()
    except Exception:
        pass
    # aggressive trim if enabled
    try:
        if args.enable_decompute:
            DecomputeManager.force_trim()
    except Exception:
        pass
    free_memory_trim()
    rss_before = measure_memory_rss()

    # Latency distribution (baseline)
    if args.skip_latency:
        lat_base = {"p50": 0.0, "p90": 0.0, "mean": 0.0, "std": 0.0, "cv": 0.0, "runs": 0.0}
    else:
        lat_base = measure_latency_distribution(
            model, tok, texts, max_length=args.max_length, warmup=args.latency_warmup, runs=args.latency_runs
        )

    # Energy (baseline)
    if args.skip_energy:
        energy_base = types.SimpleNamespace(total_flops=0.0, dyn_energy_j=0.0, landauer_lower_j=0.0, latency_s=0.0, tokens=0)
    else:
        energy_base = measure_energy(model, tok, texts, max_length=args.max_length, T_K=args.energy_T_K)

    # Apply reversible/PDM proxies on a copy pathway before optional compile
    patterns = [p.strip() for p in args.patterns.split(',') if p.strip()]
    reversible_report: Dict[str, Tuple[int, int, int]] = {}
    pdm_report: Dict[str, float] = {}
    functor_mc_report: Dict[str, float] = {}
    phase4_report: Dict[str, float] = {}
    osc_report: Dict[str, object] = {}
    cauchy_report: Dict[str, float] = {}
    morse_report: Dict[str, object] = {}
    ep_report: Dict[str, float] = {}
    holomorphic_report: Dict[str, float] = {}

    model_mod = model
    osc = None
    # Build per-proxy patterns
    cauchy_patterns = [p.strip() for p in (args.cauchy_patterns.split(',') if args.cauchy_patterns else args.patterns.split(',')) if p.strip()]
    morse_patterns = [p.strip() for p in (args.morse_patterns.split(',') if args.morse_patterns else args.patterns.split(',')) if p.strip()]

    # Apply Phase 2 (Morse) first on original weights
    if args.use_morse:
        try:
            morse_report = morse_skeletonize(
                model_mod,
                keep_frac=float(args.morse_keep_frac),
                name_patterns=morse_patterns,
                skip_lm_head=bool(args.skip_lm_head),
            )
        except Exception:
            morse_report = {}
    # Then Phase 1 (Cauchy) to compress MLP-like layers
    if args.use_cauchy:
        try:
            cauchy_report = replace_mlp_with_cauchy(
                model_mod,
                rank=int(args.cauchy_rank),
                name_patterns=cauchy_patterns,
                skip_lm_head=bool(args.skip_lm_head),
            )
        except Exception:
            cauchy_report = {"modules": 0.0, "orig_params": 0.0, "coeff_params": 0.0, "params_ratio": 1.0, "rank": float(args.cauchy_rank)}
    # Holomorphic substitution (collapse parameterization via Fourier/Cauchy basis)
    if args.use_holomorphic:
        try:
            holo_patterns = [p.strip() for p in (args.holo_patterns.split(',') if args.holo_patterns else args.patterns.split(',')) if p.strip()]
            holomorphic_report = replace_mlp_with_holomorphic(
                model_mod,
                rank=int(args.holo_rank),
                name_patterns=holo_patterns,
                skip_lm_head=bool(args.skip_lm_head),
            )
            if int(args.holo_calib_steps) > 0:
                try:
                    _ = calibrate_holomorphic(
                        model_mod,
                        tok,
                        texts,
                        steps=int(args.holo_calib_steps),
                        lr=5e-2,
                        mdl_lambda=float(args.holo_mdl_lambda),
                        max_length=args.max_length,
                    )
                except Exception:
                    pass
        except Exception:
            holomorphic_report = {"modules": 0.0, "orig_params": 0.0, "coeff_params": 0.0, "params_ratio": 1.0, "rank": float(args.holo_rank)}
    if args.use_reversible:
        reversible_report = replace_with_reversible(
            model_mod,
            rank=int(args.rev_rank),
            name_patterns=patterns,
            skip_lm_head=bool(args.skip_lm_head),
        )
        if args.mdl_in_loss and int(args.mdl_in_loss_steps) > 0:
            try:
                calibrate_reversible_mdl(
                    model_mod,
                    tok,
                    texts=texts,
                    steps=int(args.mdl_in_loss_steps),
                    lr=5e-2,
                    lambda_phys=float(args.rev_lambda),
                    mdl_lambda=float(args.mdl_in_loss_lambda),
                    max_length=args.max_length,
                )
            except Exception:
                pass
        elif int(args.rev_steps) > 0:
            try:
                calibrate_reversible(
                    model_mod,
                    tok,
                    texts=texts,
                    steps=int(args.rev_steps),
                    lr=5e-2,
                    lambda_phys=float(args.rev_lambda),
                    max_length=args.max_length,
                )
            except Exception:
                pass

    if args.use_pdm:
        try:
            pdm_report = pdm_eval_lm_head(
                model_mod,
                tok,
                texts,
                bits=int(args.pdm_bits),
                topk=int(args.pdm_topk),
                max_length=args.max_length,
            )
        except Exception:
            pdm_report = {"pdm_cos": 0.0, "pdm_time_s": 0.0, "pdm_topk": float(args.pdm_topk), "bits": float(args.pdm_bits)}

    if args.use_functor_mc:
        try:
            functor_mc_report = verify_functor_mapping(
                model_mod,
                tok,
                texts,
                name_patterns=patterns,
                max_length=args.max_length,
                max_rows=int(args.functor_max_rows),
                max_samples=int(args.functor_max_samples),
            )
        except Exception:
            functor_mc_report = {"verified_modules": 0.0, "tested_rows_total": 0.0, "mean_abs_diff": 0.0, "max_abs_diff": 0.0}

    # Phase 3 hooks and metrics/calibration
    osc_mgr = None
    if args.use_osc or args.use_ep:
        try:
            osc_mgr = OscillatoryHooks(alpha=float(args.osc_alpha), freq=float(args.osc_freq))
            osc_mgr.attach(model_mod, name_patterns=patterns, skip_lm_head=bool(args.skip_lm_head))
        except Exception:
            osc_mgr = None
    if args.use_osc and osc_mgr is not None:
        try:
            cos_v, kl_v = interference_metrics(model_mod, tok, texts, manager=osc_mgr, delta_phi=float(args.interf_dphi), max_length=args.max_length)
            osc_report.update({"interference_cos": float(cos_v), "interference_sym_kl": float(kl_v), "phase_coherence": float(osc_mgr.phase_coherence())})
        except Exception:
            pass
    if args.use_ep and (osc_mgr is not None) and int(args.ep_steps) > 0:
        try:
            ep_report = equilibrium_propagation_tune(
                model_mod,
                tok,
                texts,
                manager=osc_mgr,
                steps=int(args.ep_steps),
                lr=float(args.ep_lr),
                beta=float(args.ep_beta),
                max_length=args.max_length,
            )
        except Exception:
            ep_report = {"steps": 0.0}

    # Compile kernels (optional)
    compiled = None
    compiled_failed = False
    if args.use_compile:
        compiled = compile_model_guarded(model_mod, backend=args.compile_backend)
        # Post-compile metrics guarded at runtime (some backends compile lazily)
        try:
            t1 = time.perf_counter()
            if args.osc_infer:
                infer_mgr = OscillatoryHooks(alpha=float(args.osc_alpha), freq=float(args.osc_freq))
                try:
                    infer_mgr.attach(compiled, name_patterns=patterns, skip_lm_head=bool(args.skip_lm_head))
                except Exception:
                    pass
                comp_nll, _ = _eval_nll_osc_interf(
                    compiled, tok, texts, manager=infer_mgr,
                    passes=int(args.osc_passes), dphi=float(args.osc_dphi), max_length=args.max_length,
                    agg_mode=str(args.osc_agg), trim_each_prompt=bool(args.trim_each_prompt)
                )
                try:
                    infer_mgr.detach()
                except Exception:
                    pass
            else:
                comp_nll, _ = _eval_nll(compiled, tok, texts, max_length=args.max_length, trim_each_prompt=bool(args.trim_each_prompt))
            comp_time = time.perf_counter() - t1
            if args.skip_latency:
                lat_comp = {"p50": 0.0, "p90": 0.0, "mean": 0.0, "std": 0.0, "cv": 0.0, "runs": 0.0}
            else:
                lat_comp = measure_latency_distribution(
                    compiled, tok, texts, max_length=args.max_length, warmup=args.latency_warmup, runs=args.latency_runs
                )
            if args.skip_energy:
                energy_comp = types.SimpleNamespace(total_flops=0.0, dyn_energy_j=0.0, landauer_lower_j=0.0, latency_s=0.0, tokens=0)
            else:
                energy_comp = measure_energy(compiled, tok, texts, max_length=args.max_length, T_K=args.energy_T_K)
        except Exception:
            compiled_failed = True
            comp_nll = base_nll
            comp_time = base_time
            lat_comp = lat_base
            energy_comp = energy_base
    else:
        # Evaluate modified model without compilation
        t1 = time.perf_counter()
        if args.osc_infer:
            infer_mgr = OscillatoryHooks(alpha=float(args.osc_alpha), freq=float(args.osc_freq))
            try:
                infer_mgr.attach(model_mod, name_patterns=patterns, skip_lm_head=bool(args.skip_lm_head))
            except Exception:
                pass
            comp_nll, _ = _eval_nll_osc_interf(
                model_mod, tok, texts, manager=infer_mgr,
                passes=int(args.osc_passes), dphi=float(args.osc_dphi), max_length=args.max_length,
                agg_mode=str(args.osc_agg), trim_each_prompt=bool(args.trim_each_prompt)
            )
            try:
                infer_mgr.detach()
            except Exception:
                pass
        else:
            comp_nll, _ = _eval_nll(model_mod, tok, texts, max_length=args.max_length, trim_each_prompt=bool(args.trim_each_prompt))
        comp_time = time.perf_counter() - t1
        if args.skip_latency:
            lat_comp = {"p50": 0.0, "p90": 0.0, "mean": 0.0, "std": 0.0, "cv": 0.0, "runs": 0.0}
        else:
            lat_comp = measure_latency_distribution(
                model_mod, tok, texts, max_length=args.max_length, warmup=args.latency_warmup, runs=args.latency_runs
            )
        if args.skip_energy:
            energy_comp = types.SimpleNamespace(total_flops=0.0, dyn_energy_j=0.0, landauer_lower_j=0.0, latency_s=0.0, tokens=0)
        else:
            energy_comp = measure_energy(model_mod, tok, texts, max_length=args.max_length, T_K=args.energy_T_K)

    # Phase 4 metrics (optional)
    try:
        if args.use_solomonoff:
            phase4_report["solomonoff_mixed_nll"] = float(solomonoff_mixed_nll(model_mod, tok, texts, gamma=float(args.gamma), max_length=args.max_length))
    except Exception:
        pass
    try:
        if args.use_topos_proxy:
            phase4_report["comp_consistency"] = float(compositional_consistency(model_mod, tok, texts, max_length=args.max_length))
    except Exception:
        pass
    try:
        if args.use_hyper_profile:
            hp = hyperbolic_profile(model_mod, tok, texts, max_length=args.max_length)
            phase4_report.update(hp)
    except Exception:
        pass
    try:
        if args.use_mdl:
            phase4_report["mdl_surrogate_nll"] = float(mdl_surrogate_nll(model_mod, tok, texts, mdl_lambda=float(args.mdl_lambda), max_length=args.max_length))
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass
    # release compiled graph before measuring RSS if present
    try:
        compiled = None
        osc_mgr = None
    except Exception:
        pass
    # aggressive trim if enabled
    try:
        if args.enable_decompute:
            DecomputeManager.force_trim()
    except Exception:
        pass
    free_memory_trim()
    rss_after = measure_memory_rss()

    # Adiabatic proxy on compiled energy
    adiabatic_eta = float(max(0.0, min(1.0, args.adiabatic_eta)))
    adiabatic_recovered_j = float(energy_comp.dyn_energy_j) * adiabatic_eta
    adiabatic_residual_j = float(energy_comp.dyn_energy_j) - adiabatic_recovered_j

    results = {
        "model": args.model,
        "seed": args.seed,
        "use_compile": bool(args.use_compile),
        "compile_backend": args.compile_backend,
        "baseline": {
            "nll": base_nll,
            "eval_time_s": base_time,
            "params": base_params,
            "rss_bytes": rss_before,
            "latency": lat_base,
            "energy": {
                "total_flops": energy_base.total_flops,
                "dyn_energy_j": energy_base.dyn_energy_j,
                "landauer_lower_j": energy_base.landauer_lower_j,
                "latency_s": energy_base.latency_s,
                "tokens": energy_base.tokens,
            },
        },
        "compiled": {
            "nll": comp_nll,
            "eval_time_s": comp_time,
            "params": base_params,
            "rss_bytes": rss_after,
            "latency": lat_comp,
            "energy": {
                "total_flops": energy_comp.total_flops,
                "dyn_energy_j": energy_comp.dyn_energy_j,
                "landauer_lower_j": energy_comp.landauer_lower_j,
                "latency_s": energy_comp.latency_s,
                "tokens": energy_comp.tokens,
                "adiabatic_eta": adiabatic_eta,
                "adiabatic_recovered_j": adiabatic_recovered_j,
                "adiabatic_residual_j": adiabatic_residual_j,
            },
            "failed": compiled_failed,
        },
        "reversible_report": reversible_report,
        "pdm_report": pdm_report,
        "functor_mc_report": functor_mc_report,
        "phase4_report": phase4_report,
        "osc_report": osc_report,
        "cauchy_report": cauchy_report,
        "morse_report": morse_report,
        "holomorphic_report": holomorphic_report,
        "ep_report": ep_report,
        "timestamp": int(time.time()),
    }

    out_path = args.results_json
    if out_path is None:
        ts = int(time.time())
        out_dir = os.path.join("quality", "phase5_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"phase5_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Aggregate helper values
    morse_pruned_total = 0
    try:
        for v in morse_report.values():
            if isinstance(v, dict) and "pruned" in v:
                morse_pruned_total += int(v.get("pruned", 0))
    except Exception:
        morse_pruned_total = 0

    summary = {
        "nll_delta": float(comp_nll - base_nll),
        "eval_time_ratio": float(comp_time / max(1e-9, base_time)),
        "rss_delta_bytes": int(rss_after - rss_before),
        "latency_p50_ratio": float(lat_comp.get("p50", 0.0) / max(1e-9, lat_base.get("p50", 0.0))),
        "energy_dyn_ratio": float(results["compiled"]["energy"]["dyn_energy_j"]) / max(1e-12, results["baseline"]["energy"]["dyn_energy_j"]),
        "landauer_compiled_j": float(results["compiled"]["energy"]["landauer_lower_j"]),
        "adiabatic_eta": adiabatic_eta,
        "reversible_layers": int(len(reversible_report)) if reversible_report else 0,
        "pdm_cos": float(pdm_report.get("pdm_cos", 0.0)) if pdm_report else 0.0,
        "functor_verified_modules": float(functor_mc_report.get("verified_modules", 0.0)) if functor_mc_report else 0.0,
        "functor_mean_abs_diff": float(functor_mc_report.get("mean_abs_diff", 0.0)) if functor_mc_report else 0.0,
        "osc_coherence": float(osc_report.get("phase_coherence", 0.0)) if osc_report else 0.0,
        "interf_cos": float(osc_report.get("interference_cos", 0.0)) if osc_report else 0.0,
        "interf_kl": float(osc_report.get("interference_sym_kl", 0.0)) if osc_report else 0.0,
        "solomonoff_mixed_nll": float(phase4_report.get("solomonoff_mixed_nll", 0.0)) if phase4_report else 0.0,
        "comp_consistency": float(phase4_report.get("comp_consistency", 0.0)) if phase4_report else 0.0,
        "hyper_over_euclid": float(phase4_report.get("hyper_over_euclid", 0.0)) if phase4_report else 0.0,
        "mdl_surrogate_nll": float(phase4_report.get("mdl_surrogate_nll", 0.0)) if phase4_report else 0.0,
        "cauchy_params_ratio": float(cauchy_report.get("params_ratio", 0.0)) if cauchy_report else 0.0,
        "morse_pruned_total": int(morse_pruned_total),
        "ep_loss_delta": float(ep_report.get("loss_delta", 0.0)) if ep_report else 0.0,
        "results_path": out_path,
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
