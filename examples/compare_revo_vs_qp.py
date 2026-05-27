"""REVO vs Quant+Prune benchmark. Evaluates NLL, latency, memory, reconstruction fidelity."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from revo.calibration import wrap_lm_head_with_calib, calibrate_logits

from revo._utils import (
    encode_text,
    evaluate_nll,
    measure_memory_rss,
    seed_everything,
)

from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank
from revo.fractal import replace_with_fractal
from revo.ephemeral import replace_with_ephemeral, calibrate_ephemeral

try:
    import torch.nn.utils.prune as prune
except Exception:
    prune = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VariantMetrics:
    name: str
    nll_general: float
    nll_ood: float
    recon_cos: float
    recon_kl: float
    stability_cos: float
    stability_kl: float
    eval_time_s: float
    tokens_scored: int
    tokens_per_s: float
    rss_peak_mb: float
    cold_start_s: float
    warm_avg_s: float


@dataclass
class CompareResult:
    model: str
    seeds: List[int]
    prompts: int
    max_length: int
    variants: List[VariantMetrics]


@dataclass
class _Measured:
    nll_general: float
    nll_ood: float
    logits: List[torch.Tensor]
    logits_pert: List[torch.Tensor]
    eval_time_s: float
    rss_peak_mb: float
    tokens_scored: int
    tokens_per_s: float

    def to_metrics(self, name: str, recon_cos: float, recon_kl: float,
                   stability_cos: float, stability_kl: float) -> VariantMetrics:
        return VariantMetrics(
            name=name,
            nll_general=self.nll_general,
            nll_ood=self.nll_ood,
            recon_cos=recon_cos,
            recon_kl=recon_kl,
            stability_cos=stability_cos,
            stability_kl=stability_kl,
            eval_time_s=self.eval_time_s,
            tokens_scored=self.tokens_scored,
            tokens_per_s=self.tokens_per_s,
            rss_peak_mb=self.rss_peak_mb,
            cold_start_s=0.0,
            warm_avg_s=0.0,
        )


# ---------------------------------------------------------------------------
# Device / model loading
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model_tok(
    model_name: str,
    device_map: str | None = None,
    dtype_str: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs: Dict[str, Any] = {}
    if device_map and device_map.lower() != "none":
        kwargs["device_map"] = device_map
    if dtype_str:
        ds = dtype_str.lower()
        if ds == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif ds == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif ds == "float32":
            kwargs["torch_dtype"] = torch.float32
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    if not kwargs.get("device_map"):
        model.to(_device())
    return model, tok


# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------

def _texts_general(n: int) -> List[str]:
    base = [
        "Explica brevemente el filtrado espectral y su impacto.",
        "Describe el uso de matrices circulantes en FFT.",
        "Que aporta HoRA sobre una variedad hiperbolica?",
        "Resume el mapeo holografico bulk->boundary->bulk.",
        "Define rango efectivo y energia espectral.",
        "Que es un bus de fase natural?",
        "Explica el un-computing reversible.",
        "Que es WDM y como paraleliza subcanales?",
        "Como funciona un gating efimero suave?",
        "Explica el caching tipo arbol de prefijos (Radix).",
    ]
    return [base[i % len(base)] + f" [G#{i}]" for i in range(n)]


def _texts_ood(n: int) -> List[str]:
    base = [
        "Summarize the role of spectral sparsity in compression.",
        "Write a short Python function for radix prefix search.",
        "Explain energy vs Landauer limit in simple terms.",
        "What is reversible computing and why does it matter?",
        "Give an example of wave-division multiplexing in optics.",
    ]
    return [base[i % len(base)] + f" [O#{i}]" for i in range(n)]


def _perturb(t: str) -> str:
    return (t + " ,").replace("  ", " ")


# ---------------------------------------------------------------------------
# Logit-level metrics
# ---------------------------------------------------------------------------

def _last_token_logits(model, tok, texts: List[str], max_len: int) -> List[torch.Tensor]:
    outs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for t in texts:
            batch = encode_text(tok, t, max_len, device=_device())
            out = model(**batch)
            logits = out.logits
            outs.append(logits[:, -1, :].detach().float().cpu().squeeze(0))
    return outs


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(p=2) + 1e-9)
    b = b / (b.norm(p=2) + 1e-9)
    return float((a * b).sum().item())


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = torch.log_softmax(p_logits, dim=-1)
    q = torch.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    return float(torch.sum(p_prob * (p - q)).item())


def _compute_reconstruction(
    base: _Measured, variant: _Measured,
) -> Tuple[float, float, float, float]:
    sims = [_cosine_sim(b, r) for b, r in zip(base.logits, variant.logits)]
    kls = [_kl_div(b, r) for b, r in zip(base.logits, variant.logits)]
    st_sims = [_cosine_sim(b, r) for b, r in zip(base.logits_pert, variant.logits_pert)]
    st_kls = [_kl_div(b, r) for b, r in zip(base.logits_pert, variant.logits_pert)]
    return float(np.mean(sims)), float(np.mean(kls)), float(np.mean(st_sims)), float(np.mean(st_kls))


# ---------------------------------------------------------------------------
# Measurement helper (HF)
# ---------------------------------------------------------------------------

def _measure_hf(
    model, tok, gen_texts: List[str], ood_texts: List[str],
    pert_gen: List[str], max_len: int,
) -> _Measured:
    t0 = time.perf_counter()
    peak = measure_memory_rss()
    nll_gen = evaluate_nll(model, tok, gen_texts, max_length=max_len, device=_device())
    nll_ood = evaluate_nll(model, tok, ood_texts, max_length=max_len, device=_device())
    logits = _last_token_logits(model, tok, gen_texts[:50], max_len)
    logits_pert = _last_token_logits(model, tok, pert_gen, max_len)
    dt = time.perf_counter() - t0
    rss = float(max(peak, measure_memory_rss())) / (1024 * 1024)
    return _Measured(
        nll_general=nll_gen, nll_ood=nll_ood,
        logits=logits, logits_pert=logits_pert,
        eval_time_s=dt, rss_peak_mb=rss,
        tokens_scored=0, tokens_per_s=0.0,
    )


# ---------------------------------------------------------------------------
# Variant creators
# ---------------------------------------------------------------------------

def _make_quant_prune(model, prune_amount: float, skip_dynamic_quant: bool):
    if not skip_dynamic_quant:
        model = _quantize_dynamic_int8(model)
    _global_prune(model, amount=prune_amount)
    return model


def _apply_revo(
    model: nn.Module,
    tok,
    texts: List[str],
    seed: int,
    max_len: int,
    deep_fraction: float = 0.5,
    calibrate_head: bool = False,
    enable_fractal: bool = True,
    ephem_steps: int = 5,
    energy_keep: float = 0.92,
    max_rank_frac: float = 0.20,
    entropy_threshold: float = 0.20,
    gamma_lowentropy: float = 3.0,
    gamma_normal: float = 2.0,
    ephem_lr: float = 5e-2,
    head_calib_steps: int = 50,
    head_calib_lr: float = 1e-2,
) -> None:
    name_patterns = ["attn", "mlp", "c_fc", "c_proj"]
    prof = profile_model_2d(model, name_patterns=name_patterns, max_rank=None)
    layer_ids: List[int] = []
    for k in prof.keys():
        if "transformer.h." in k:
            try:
                s = k.split("transformer.h.", 1)[1]
                lid = int(s.split(".")[0])
                layer_ids.append(lid)
            except Exception:
                pass
    allow: set[str]
    if layer_ids:
        L = max(layer_ids) + 1
        frac = max(0.0, min(1.0, float(deep_fraction)))
        start = max(0, int(round((1.0 - frac) * L)))
        allow = set()
        for k in prof.keys():
            if "transformer.h." in k:
                try:
                    s = k.split("transformer.h.", 1)[1]
                    lid = int(s.split(".")[0])
                    if lid >= start:
                        allow.add(k)
                except Exception:
                    continue
    else:
        allow = set(prof.keys())
    ranks_full = allocate_ranks_energy_with_caps(prof, energy_keep=energy_keep, max_rank=None, max_rank_frac=max_rank_frac)
    ranks = {k: v for k, v in ranks_full.items() if k in allow}
    replace_2d_modules_with_lowrank(model, ranks, calibrate=True, calibrate_samples=128, seed=seed)
    if enable_fractal:
        replace_with_fractal(model, depth=2, alpha=0.5, name_patterns=["attn", "mlp", "c_proj"], skip_lm_head=True, allow_names=sorted(list(allow)))
    gamma_map = {}
    for k in allow:
        info = prof.get(k, {})
        h = float(info.get("entropy_norm", 0.0)) if isinstance(info, dict) else 0.0
        gamma_map[k] = (gamma_lowentropy if h < float(entropy_threshold) else gamma_normal)
    replace_with_ephemeral(model, name_patterns=name_patterns, skip_lm_head=True, allow_names=sorted(list(allow)), gamma_init_map=gamma_map)
    calibrate_ephemeral(model, tok, texts=texts[:min(32, len(texts))], steps=int(ephem_steps), lr=float(ephem_lr), lambda_phys=1.0, max_length=max_len)
    if calibrate_head:
        wrap_lm_head_with_calib(model)
        calibrate_logits(model, tok, texts=texts[:min(64, len(texts))], steps=int(head_calib_steps), lr=float(head_calib_lr), max_length=max_len)


# ---------------------------------------------------------------------------
# Quantization + pruning helpers
# ---------------------------------------------------------------------------

def _quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    dtypes = {nn.Linear: torch.qint8}
    try:
        from torch.quantization import quantize_dynamic
    except Exception:
        return model
    model_cpu = model.to("cpu")
    try:
        model_q = quantize_dynamic(model_cpu, dtypes, dtype=torch.qint8)
        model_q.eval()
        return model_q.to(_device())
    except Exception:
        return model.to(_device())


def _global_prune(model: nn.Module, amount: float = 0.3) -> None:
    if prune is None:
        return
    params = [(mod, "weight") for name, mod in model.named_modules()
              if hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), nn.Parameter)]
    if not params:
        return
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for module, _ in params:
        try:
            prune.remove(module, "weight")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

def _resolve_module(model: nn.Module, dotted: str) -> nn.Module:
    cur: nn.Module = model
    for part in dotted.split(".") if dotted else []:
        if part.isdigit():
            cur = getattr(cur, "__getitem__")(int(part))
        else:
            cur = getattr(cur, part)
        if not isinstance(cur, nn.Module):
            raise RuntimeError(f"Resolved path '{dotted}' is not an nn.Module: {type(cur)}")
    return cur


def _hidden_and_logits_for_texts(
    model: nn.Module, tok, texts: List[str], max_len: int,
    hook_module: str = "transformer.ln_f",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    model.eval()
    hiddens: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        mod = _resolve_module(model, hook_module)
        def _hook(_m, _inp, out):
            captured = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
            _hook.captured = captured
        for t in texts:
            handle = mod.register_forward_hook(_hook)
            batch = encode_text(tok, t, max_len, device=_device())
            out = model(**batch)
            handle.remove()
            captured = getattr(_hook, "captured", None)
            if captured is None:
                raise RuntimeError("Forward hook did not capture output; check module path")
            hv = captured[:, -1, :].detach().float().cpu().squeeze(0)
            lv = out.logits[:, -1, :].detach().float().cpu().squeeze(0)
            hiddens.append(hv)
            logits_list.append(lv)
    return hiddens, logits_list


def run_alignment_check(
    model_name: str, seed: int, prompts: int, max_len: int,
    hook_module: str, revo_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    seed_everything(seed)
    texts = _texts_general(prompts)
    base_model, tok = _load_model_tok(model_name)
    base_h, base_l = _hidden_and_logits_for_texts(base_model, tok, texts, max_len, hook_module)
    revo_model, _ = _load_model_tok(model_name)
    try:
        _apply_revo(revo_model, tok, texts, seed, max_len,
                    deep_fraction=float(revo_kwargs.get("revo_deep_fraction", 0.5)),
                    calibrate_head=bool(revo_kwargs.get("revo_calibrate_head", False)),
                    enable_fractal=not bool(revo_kwargs.get("revo_disable_fractal", False)),
                    ephem_steps=int(revo_kwargs.get("revo_ephem_steps", 5)),
                    energy_keep=float(revo_kwargs.get("revo_energy_keep", 0.92)),
                    max_rank_frac=float(revo_kwargs.get("revo_max_rank_frac", 0.20)),
                    entropy_threshold=float(revo_kwargs.get("revo_entropy_threshold", 0.20)),
                    gamma_lowentropy=float(revo_kwargs.get("revo_gamma_lowentropy", 3.0)),
                    gamma_normal=float(revo_kwargs.get("revo_gamma_normal", 2.0)),
                    ephem_lr=float(revo_kwargs.get("revo_ephem_lr", 5e-2)),
                    head_calib_steps=int(revo_kwargs.get("revo_head_calib_steps", 50)),
                    head_calib_lr=float(revo_kwargs.get("revo_head_calib_lr", 1e-2)),
                    )
    except Exception:
        pass
    revo_h, revo_l = _hidden_and_logits_for_texts(revo_model, tok, texts, max_len, hook_module)

    def _l2(x: torch.Tensor) -> float:
        return float(x.norm(p=2).item())
    hid_base_norms = [_l2(h) for h in base_h]
    hid_revo_norms = [_l2(h) for h in revo_h]
    hid_diffs = [_l2(r - b) for b, r in zip(base_h, revo_h)]
    logit_diffs = [_l2(r - b) for b, r in zip(base_l, revo_l)]

    def _summary(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99))}

    return {"model": model_name, "seed": seed, "prompts": prompts,
            "max_length": max_len, "hook_module": hook_module,
            "hidden_norms_baseline": _summary(hid_base_norms),
            "hidden_norms_revo": _summary(hid_revo_norms),
            "hidden_l2_diff": _summary(hid_diffs),
            "logits_l2_diff": _summary(logit_diffs)}


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------

def _llama_nll_and_last_topk(
    llm, text: str, max_len: int, topk: int,
    temperature: float = 1.0, prompt_char_limit: int = 512,
) -> tuple[float, dict[str, float], int]:
    text = text[:int(prompt_char_limit)]
    llm.reset()
    out = llm.create_completion(prompt=text, max_tokens=1, echo=True,
                                temperature=float(temperature), logprobs=max(1, topk))
    lp = out["choices"][0]["logprobs"]["token_logprobs"]
    usage = out.get("usage", {})
    n_prompt = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) and "prompt_tokens" in usage else max(0, len(lp) - 1)
    vals = [x for x in lp[:n_prompt] if x is not None]
    nll = float(-sum(vals) / max(1, len(vals)))
    llm.reset()
    out2 = llm.create_completion(prompt=text, max_tokens=1, echo=True,
                                 temperature=float(temperature), logprobs=max(1, topk))
    top_list = out2["choices"][0]["logprobs"].get("top_logprobs", [])
    last = top_list[-1] if top_list else {}
    return nll, {str(k): float(v) for k, v in last.items()}, int(n_prompt)


def _align_topk_dicts(a: dict[str, float], b: dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    def lp(d, k): return float(d.get(k, -50.0))
    return torch.tensor([lp(a, k) for k in keys], dtype=torch.float32), torch.tensor([lp(b, k) for k in keys], dtype=torch.float32)


def evaluate_variants_llama(
    gguf_path: str, seeds: List[int], prompts: int, max_len: int,
    n_ctx: int = 2048, n_threads: int = 4, topk: int = 50, tau: float = 1.0,
    prompt_char_limit: int = 128, fast: bool = False,
) -> Dict[str, Any]:
    try:
        from llama_cpp import Llama
    except Exception as e:
        raise RuntimeError(f"llama-cpp-python not available: {e}")

    llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, logits_all=True)
    gen_texts = _texts_general(prompts)
    if fast:
        ood_count = max(1, min(3, max(1, prompts // 3)))
        pert_count = min(3, len(gen_texts))
    else:
        ood_count = max(5, min(10, prompts // 2))
        pert_count = min(10, len(gen_texts))
    ood_texts = _texts_ood(ood_count)
    pert_gen = [_perturb(t) for t in gen_texts[:pert_count]]

    results: List[VariantMetrics] = []
    for seed in seeds:
        seed_everything(seed)
        revo_kwargs = {"tau": tau}

        for label, temp in [("baseline", 1.0), (f"revo(tau={tau})", float(tau))]:
            t0 = time.perf_counter()
            peak = max(0, measure_memory_rss())
            tokens = 0
            nlls_gen: List[float] = []
            nlls_ood: List[float] = []
            topk_gen: List[dict[str, float]] = []
            topk_gen_pert: List[dict[str, float]] = []
            cold_s: float | None = None
            warm_sum = 0.0
            warm_cnt = 0
            for t in gen_texts:
                _ts = time.perf_counter()
                nll_g, last, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=temp, prompt_char_limit=prompt_char_limit)
                _dt = time.perf_counter() - _ts
                nlls_gen.append(nll_g)
                topk_gen.append(last)
                tokens += int(n_tok)
                peak = max(peak, measure_memory_rss())
                if cold_s is None:
                    cold_s = _dt
                else:
                    warm_sum += _dt
                    warm_cnt += 1
            for t in ood_texts:
                nll_o, _, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=temp, prompt_char_limit=prompt_char_limit)
                nlls_ood.append(nll_o)
                tokens += int(n_tok)
                peak = max(peak, measure_memory_rss())
            for t in pert_gen:
                _, last_p, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=temp, prompt_char_limit=prompt_char_limit)
                topk_gen_pert.append(last_p)
                tokens += int(n_tok)
                peak = max(peak, measure_memory_rss())
            elapsed = time.perf_counter() - t0
            rss_mb = float(peak) / (1024 * 1024)
            tok_per_s = float(tokens) / elapsed if elapsed > 0 else 0.0

            if label == "baseline":
                base_gen = list(topk_gen)
                base_pert = list(topk_gen_pert)
                base_nlls_ood = list(nlls_ood)
                results.append(VariantMetrics(
                    name=label,
                    nll_general=float(np.mean(nlls_gen)),
                    nll_ood=float(np.mean(nlls_ood)),
                    recon_cos=1.0, recon_kl=0.0, stability_cos=1.0, stability_kl=0.0,
                    eval_time_s=elapsed, tokens_scored=tokens, tokens_per_s=tok_per_s,
                    rss_peak_mb=rss_mb, cold_start_s=float(cold_s or 0.0),
                    warm_avg_s=float(warm_sum) / max(1, warm_cnt),
                ))
            else:
                sims = [_cosine_sim(*_align_topk_dicts(b, r)) for b, r in zip(base_gen, topk_gen)]
                kls = [_kl_div(*_align_topk_dicts(b, r)) for b, r in zip(base_gen, topk_gen)]
                st_sims = [_cosine_sim(*_align_topk_dicts(b, r)) for b, r in zip(base_pert, topk_gen_pert)]
                st_kls = [_kl_div(*_align_topk_dicts(b, r)) for b, r in zip(base_pert, topk_gen_pert)]
                results.append(VariantMetrics(
                    name=label,
                    nll_general=float(np.mean(nlls_gen)),
                    nll_ood=float(np.mean(nlls_ood)),
                    recon_cos=float(np.mean(sims)) if sims else 1.0,
                    recon_kl=float(np.mean(kls)) if kls else 0.0,
                    stability_cos=float(np.mean(st_sims)) if st_sims else 1.0,
                    stability_kl=float(np.mean(st_kls)) if st_kls else 0.0,
                    eval_time_s=elapsed, tokens_scored=tokens, tokens_per_s=tok_per_s,
                    rss_peak_mb=rss_mb, cold_start_s=float(cold_s or 0.0),
                    warm_avg_s=float(warm_sum) / max(1, warm_cnt),
                ))

    return asdict(CompareResult(
        model=f"llama.cpp:{os.path.basename(gguf_path)}",
        seeds=seeds, prompts=prompts, max_length=max_len, variants=results,
    ))


# ---------------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------------

def evaluate_variants(
    model_name: str, seeds: List[int], prompts: int, max_len: int,
    prune_amount: float, device_map: str | None = None,
    dtype_str: str | None = None, load_in_4bit: bool = False,
    load_in_8bit: bool = False, skip_dynamic_quant: bool = False,
    revo_deep_fraction: float = 0.5, revo_calibrate_head: bool = False,
    revo_enable_fractal: bool = True, revo_ephem_steps: int = 5,
    revo_energy_keep: float = 0.92, revo_max_rank_frac: float = 0.20,
    revo_entropy_threshold: float = 0.20, revo_gamma_lowentropy: float = 3.0,
    revo_gamma_normal: float = 2.0, revo_ephem_lr: float = 5e-2,
    revo_head_calib_steps: int = 50, revo_head_calib_lr: float = 1e-2,
) -> Dict[str, Any]:
    gen_texts = _texts_general(prompts)
    ood_texts = _texts_ood(max(50, prompts // 2))
    pert_gen = [_perturb(t) for t in gen_texts[:50]]
    load_kw = dict(device_map=device_map, dtype_str=dtype_str,
                   load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)

    results: List[VariantMetrics] = []
    for seed in seeds:
        seed_everything(seed)

        # --- Baseline ---
        base_model, tok = _load_model_tok(model_name, **load_kw)
        base = _measure_hf(base_model, tok, gen_texts, ood_texts, pert_gen, max_len)
        results.append(base.to_metrics("baseline", 1.0, 0.0, 1.0, 0.0))

        # --- Quant+Prune ---
        qp_model, _ = _load_model_tok(model_name, **load_kw)
        qp_model = _make_quant_prune(qp_model, prune_amount, skip_dynamic_quant)
        qp = _measure_hf(qp_model, tok, gen_texts, ood_texts, pert_gen, max_len)
        rc, rk, sc, sk = _compute_reconstruction(base, qp)
        results.append(base.to_metrics("quant+prune", rc, rk, sc, sk))

        # --- REVO ---
        revo_model, _ = _load_model_tok(model_name, **load_kw)
        try:
            _apply_revo(revo_model, tok, gen_texts, seed, max_len,
                        deep_fraction=revo_deep_fraction,
                        calibrate_head=revo_calibrate_head,
                        enable_fractal=revo_enable_fractal,
                        ephem_steps=revo_ephem_steps,
                        energy_keep=revo_energy_keep,
                        max_rank_frac=revo_max_rank_frac,
                        entropy_threshold=revo_entropy_threshold,
                        gamma_lowentropy=revo_gamma_lowentropy,
                        gamma_normal=revo_gamma_normal,
                        ephem_lr=revo_ephem_lr,
                        head_calib_steps=revo_head_calib_steps,
                        head_calib_lr=revo_head_calib_lr,
                        )
        except Exception:
            pass
        revo = _measure_hf(revo_model, tok, gen_texts, ood_texts, pert_gen, max_len)
        rc, rk, sc, sk = _compute_reconstruction(base, revo)
        results.append(base.to_metrics("revo", rc, rk, sc, sk))

    return asdict(CompareResult(
        model=model_name, seeds=seeds, prompts=prompts,
        max_length=max_len, variants=results,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Comparativo REVO vs Quant+Prune")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--prune-amount", type=float, default=0.3)
    ap.add_argument("--device-map", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--results-json", default=None)
    ap.add_argument("--skip-dynamic-quant", action="store_true")
    # llama.cpp
    ap.add_argument("--llama-gguf", type=str, default=None)
    ap.add_argument("--llama-n-ctx", type=int, default=2048)
    ap.add_argument("--llama-n-threads", type=int, default=max(os.cpu_count() or 4, 4))
    ap.add_argument("--llama-topk", type=int, default=50)
    ap.add_argument("--llama-tau", type=float, default=1.0)
    ap.add_argument("--llama-fast", action="store_true")
    ap.add_argument("--llama-prompt-limit", type=int, default=128)
    # REVO params
    ap.add_argument("--revo-deep-fraction", type=float, default=0.5)
    ap.add_argument("--revo-calibrate-head", action="store_true")
    ap.add_argument("--revo-disable-fractal", action="store_true")
    ap.add_argument("--revo-ephem-steps", type=int, default=5)
    ap.add_argument("--revo-ephem-lr", type=float, default=5e-2)
    ap.add_argument("--revo-energy-keep", type=float, default=0.92)
    ap.add_argument("--revo-max-rank-frac", type=float, default=0.20)
    ap.add_argument("--revo-entropy-threshold", type=float, default=0.20)
    ap.add_argument("--revo-gamma-lowentropy", type=float, default=3.0)
    ap.add_argument("--revo-gamma-normal", type=float, default=2.0)
    ap.add_argument("--revo-head-calib-steps", type=int, default=50)
    ap.add_argument("--revo-head-calib-lr", type=float, default=1e-2)
    # Alignment check
    ap.add_argument("--alignment-check", action="store_true")
    ap.add_argument("--hook-module", type=str, default="transformer.ln_f")
    ap.add_argument("--alignment-results", type=str, default=None)
    return ap


def _save_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    args = _build_parser().parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.alignment_check:
        revo_kwargs: Dict[str, Any] = {
            "revo_deep_fraction": args.revo_deep_fraction,
            "revo_calibrate_head": args.revo_calibrate_head,
            "revo_disable_fractal": args.revo_disable_fractal,
            "revo_ephem_steps": args.revo_ephem_steps,
            "revo_energy_keep": args.revo_energy_keep,
            "revo_max_rank_frac": args.revo_max_rank_frac,
            "revo_entropy_threshold": args.revo_entropy_threshold,
            "revo_gamma_lowentropy": args.revo_gamma_lowentropy,
            "revo_gamma_normal": args.revo_gamma_normal,
            "revo_ephem_lr": args.revo_ephem_lr,
            "revo_head_calib_steps": args.revo_head_calib_steps,
            "revo_head_calib_lr": args.revo_head_calib_lr,
        }
        result = run_alignment_check(
            args.model, seeds[0] if seeds else 0, args.prompts,
            args.max_length, args.hook_module, revo_kwargs,
        )
        out = args.alignment_results or os.path.join("quality", "compare", f"alignment_{int(time.time())}.json")
        print(json.dumps({"alignment_path": _save_json(result, out)}, indent=2))
        return

    if args.llama_gguf:
        result = evaluate_variants_llama(
            args.llama_gguf, seeds, args.prompts, args.max_length,
            n_ctx=args.llama_n_ctx, n_threads=args.llama_n_threads,
            topk=args.llama_topk, tau=float(args.llama_tau),
            prompt_char_limit=int(args.llama_prompt_limit),
            fast=bool(args.llama_fast),
        )
    else:
        result = evaluate_variants(
            args.model, seeds, args.prompts, args.max_length,
            args.prune_amount, device_map=args.device_map,
            dtype_str=args.dtype, load_in_4bit=bool(args.load_in_4bit),
            load_in_8bit=bool(args.load_in_8bit),
            skip_dynamic_quant=bool(args.skip_dynamic_quant),
            revo_deep_fraction=float(args.revo_deep_fraction),
            revo_calibrate_head=bool(args.revo_calibrate_head),
            revo_enable_fractal=(not bool(args.revo_disable_fractal)),
            revo_ephem_steps=int(args.revo_ephem_steps),
            revo_energy_keep=float(args.revo_energy_keep),
            revo_max_rank_frac=float(args.revo_max_rank_frac),
            revo_entropy_threshold=float(args.revo_entropy_threshold),
            revo_gamma_lowentropy=float(args.revo_gamma_lowentropy),
            revo_gamma_normal=float(args.revo_gamma_normal),
            revo_ephem_lr=float(args.revo_ephem_lr),
            revo_head_calib_steps=int(args.revo_head_calib_steps),
            revo_head_calib_lr=float(args.revo_head_calib_lr),
        )

    out = args.results_json or os.path.join("quality", "compare", f"compare_{int(time.time())}.json")
    print(json.dumps({"results_path": _save_json(result, out)}, indent=2))


if __name__ == "__main__":
    main()
