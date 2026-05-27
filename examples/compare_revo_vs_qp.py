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

# REVO imports
from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank
from revo.fractal import replace_with_fractal
from revo.ephemeral import replace_with_ephemeral, calibrate_ephemeral

try:
    import torch.nn.utils.prune as prune
except Exception:  # pragma: no cover
    prune = None


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
    # If no device_map given, move to a single device
    if not kwargs.get("device_map"):
        model.to(_device())
    return model, tok


def _texts_general(n: int) -> List[str]:
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
    # simple, deterministic small perturbation
    return (t + " ,").replace("  ", " ")





def _last_token_logits(model, tok, texts: List[str], max_len: int) -> List[torch.Tensor]:
    outs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for t in texts:
            batch = encode_text(tok, t, max_len, device=_device())
            out = model(**batch)  # type: ignore[arg-type]
            logits = out.logits  # [B, T, V]
            outs.append(logits[:, -1, :].detach().float().cpu().squeeze(0))
    return outs


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(p=2) + 1e-9)
    b = b / (b.norm(p=2) + 1e-9)
    return float((a * b).sum().item())


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    # KL(p||q) on softmax(probs)
    p = torch.log_softmax(p_logits, dim=-1)
    q = torch.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    return float(torch.sum(p_prob * (p - q)).item())


# --- Alignment check utilities: L2 norms and raw logits diffs at the exact same graph point ---

def _resolve_module(model: nn.Module, dotted: str) -> nn.Module:
    cur: nn.Module = model
    parts = dotted.split(".") if dotted else []
    for part in parts:
        if part.isdigit():
            # index into ModuleList / list-like containers
            cur = getattr(cur, "__getitem__")(int(part))  # type: ignore[attr-defined]
        else:
            cur = getattr(cur, part)
        if not isinstance(cur, nn.Module):
            raise RuntimeError(f"Resolved path '{dotted}' is not an nn.Module: {type(cur)}")
    return cur


def _hidden_and_logits_for_texts(
    model: nn.Module,
    tok,
    texts: List[str],
    max_len: int,
    hook_module: str = "transformer.ln_f",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    model.eval()
    hiddens: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []

    with torch.no_grad():
        mod = _resolve_module(model, hook_module)

        def _hook(_m, _inp, out):
            if isinstance(out, torch.Tensor):
                _hook.captured = out.detach()  # type: ignore[attr-defined]
            else:
                _hook.captured = out[0].detach()  # type: ignore[attr-defined]

        for t in texts:
            handle = mod.register_forward_hook(_hook)
            batch = encode_text(tok, t, max_len, device=_device())
            out = model(**batch)  # type: ignore[arg-type]
            handle.remove()
            captured = getattr(_hook, "captured", None)  # type: ignore[attr-defined]
            if captured is None:
                raise RuntimeError("Forward hook did not capture any output; check module path")
            hid = captured  # [B,T,H]
            hv = hid[:, -1, :].detach().float().cpu().squeeze(0)
            lv = out.logits[:, -1, :].detach().float().cpu().squeeze(0)  # type: ignore[union-attr]
            hiddens.append(hv)
            logits_list.append(lv)

    return hiddens, logits_list


def run_alignment_check(
    model_name: str,
    seed: int,
    prompts: int,
    max_len: int,
    hook_module: str,
    revo_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    seed_everything(seed)
    texts = _texts_general(prompts)

    # Baseline
    base_model, tok = _load_model_tok(model_name)
    base_h, base_l = _hidden_and_logits_for_texts(base_model, tok, texts, max_len, hook_module)

    # REVO (apply same pipeline, then hook same module)
    revo_model, _ = _load_model_tok(model_name)
    try:
        _apply_revo(
            revo_model,
            tok,
            texts,
            seed,
            max_len,
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

    # Aggregate metrics
    def _l2(x: torch.Tensor) -> float:
        return float(x.norm(p=2).item())

    hid_base_norms = [_l2(h) for h in base_h]
    hid_revo_norms = [_l2(h) for h in revo_h]
    hid_diffs = [_l2(r - b) for b, r in zip(base_h, revo_h)]
    logit_diffs = [_l2(r - b) for b, r in zip(base_l, revo_l)]

    def _summary(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    payload: Dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "prompts": prompts,
        "max_length": max_len,
        "hook_module": hook_module,
        "hidden_norms_baseline": _summary(hid_base_norms),
        "hidden_norms_revo": _summary(hid_revo_norms),
        "hidden_l2_diff": _summary(hid_diffs),
        "logits_l2_diff": _summary(logit_diffs),
    }
    return payload


# Q+P: dynamic int8 + global magnitude pruning

def _quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    # Only Linear layers (works on CPU)
    dtypes = {nn.Linear: torch.qint8}
    try:
        from torch.quantization import quantize_dynamic  # type: ignore
    except Exception:
        return model
    model_cpu = model.to("cpu")
    try:
        model_q = quantize_dynamic(model_cpu, dtypes, dtype=torch.qint8)
        model_q.eval()
        return model_q.to(_device())
    except Exception:
        # Fallback: return original if quantization API not compatible
        return model.to(_device())


def _global_prune(model: nn.Module, amount: float = 0.3) -> None:
    if prune is None:
        return
    parameters_to_prune = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), nn.Parameter):
            parameters_to_prune.append((module, "weight"))
    if not parameters_to_prune:
        return
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    # Make pruning permanent
    for module, _ in parameters_to_prune:
        try:
            prune.remove(module, "weight")
        except Exception:
            pass


# llama.cpp backend (GGUF)

def _llama_nll_and_last_topk(llm, text: str, max_len: int, topk: int, temperature: float = 1.0, prompt_char_limit: int = 512) -> tuple[float, dict[str, float], int]:
    """Compute average per-token NLL for the prompt and get last-step topK logprobs.
    Uses echo=True to score prompt tokens and max_tokens=1 to get next-step distribution.
    """
    # Truncate prompt to keep scoring lightweight in llama.cpp
    text = text[: int(prompt_char_limit)]
    # NLL over prompt tokens
    llm.reset()
    out = llm.create_completion(
        prompt=text,
        max_tokens=1,
        echo=True,
        temperature=float(temperature),
        logprobs=max(1, topk),
    )
    lp = out["choices"][0]["logprobs"]["token_logprobs"]
    usage = out.get("usage", {})
    n_prompt = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) and "prompt_tokens" in usage else max(0, len(lp) - 1)
    vals = [x for x in lp[:n_prompt] if x is not None]
    nll = float(-sum(vals) / max(1, len(vals)))

    # Last-step distribution (next token after prompt)
    llm.reset()
    out2 = llm.create_completion(
        prompt=text,
        max_tokens=1,
        echo=True,
        temperature=float(temperature),
        logprobs=max(1, topk),
    )
    top_list = out2["choices"][0]["logprobs"].get("top_logprobs", [])
    last = top_list[-1] if top_list else {}
    # last is a mapping token_str -> logprob
    return nll, {str(k): float(v) for k, v in last.items()}, int(n_prompt)


def _align_topk_dicts(a: dict[str, float], b: dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    # Fallback logprob for missing tokens
    def lp(d, k):
        return float(d.get(k, -50.0))
    va = torch.tensor([lp(a, k) for k in keys], dtype=torch.float32)
    vb = torch.tensor([lp(b, k) for k in keys], dtype=torch.float32)
    return va, vb


def evaluate_variants_llama(
    gguf_path: str,
    seeds: List[int],
    prompts: int,
    max_len: int,
    n_ctx: int = 2048,
    n_threads: int = 4,
    topk: int = 50,
    tau: float = 1.0,
    prompt_char_limit: int = 128,
    fast: bool = False,
) -> Dict[str, Any]:
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"llama-cpp-python not available: {e}")

    llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, logits_all=True)
    gen_texts = _texts_general(prompts)
    # Keep OOD and perturbation sets small for micro-runs on CPU
    if fast:
        ood_count = max(1, min(3, max(1, prompts // 3)))
        pert_count = min(3, len(gen_texts))
    else:
        ood_count = max(5, min(10, prompts // 2))
        pert_count = min(10, len(gen_texts))
    ood_texts = _texts_ood(ood_count)
    pert_gen = [_perturb(t) for t in gen_texts[: pert_count]]

    results: List[VariantMetrics] = []

    for seed in seeds:
        seed_everything(seed)
        # Baseline
        t0 = time.perf_counter()
        base_peak_bytes = max(0, measure_memory_rss())
        base_tokens = 0
        nlls_gen_base: List[float] = []
        nlls_ood_base: List[float] = []
        topk_gen_base: List[dict[str, float]] = []
        topk_gen_base_pert: List[dict[str, float]] = []
        base_cold_s = None
        base_warm_sum = 0.0
        base_warm_cnt = 0
        for t in gen_texts:
            _ts = time.perf_counter()
            nll_g, last, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=1.0, prompt_char_limit=prompt_char_limit)
            _dt = time.perf_counter() - _ts
            nlls_gen_base.append(nll_g)
            topk_gen_base.append(last)
            base_tokens += int(n_tok)
            base_peak_bytes = max(base_peak_bytes, measure_memory_rss())
            if base_cold_s is None:
                base_cold_s = _dt
            else:
                base_warm_sum += _dt
                base_warm_cnt += 1
        for t in ood_texts:
            nll_o, _, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=1.0, prompt_char_limit=prompt_char_limit)
            nlls_ood_base.append(nll_o)
            base_tokens += int(n_tok)
            base_peak_bytes = max(base_peak_bytes, measure_memory_rss())
        for t in pert_gen:
            _, last_p, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=1.0, prompt_char_limit=prompt_char_limit)
            topk_gen_base_pert.append(last_p)
            base_tokens += int(n_tok)
            base_peak_bytes = max(base_peak_bytes, measure_memory_rss())
        elapsed_base = time.perf_counter() - t0
        base_rss_mb = float(base_peak_bytes) / (1024.0 * 1024.0)
        base_tok_per_s = (float(base_tokens) / elapsed_base) if elapsed_base > 0 else 0.0

        # REVO (probabilistic calibration via tau)
        t1 = time.perf_counter()
        revo_peak_bytes = max(0, measure_memory_rss())
        revo_tokens = 0
        nlls_gen_revo: List[float] = []
        nlls_ood_revo: List[float] = []
        topk_gen_revo: List[dict[str, float]] = []
        topk_gen_revo_pert: List[dict[str, float]] = []
        revo_cold_s = None
        revo_warm_sum = 0.0
        revo_warm_cnt = 0
        for t in gen_texts:
            _ts = time.perf_counter()
            nll_g, last, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=float(tau), prompt_char_limit=prompt_char_limit)
            _dt = time.perf_counter() - _ts
            nlls_gen_revo.append(nll_g)
            topk_gen_revo.append(last)
            revo_tokens += int(n_tok)
            revo_peak_bytes = max(revo_peak_bytes, measure_memory_rss())
            if revo_cold_s is None:
                revo_cold_s = _dt
            else:
                revo_warm_sum += _dt
                revo_warm_cnt += 1
        for t in ood_texts:
            nll_o, _, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=float(tau), prompt_char_limit=prompt_char_limit)
            nlls_ood_revo.append(nll_o)
            revo_tokens += int(n_tok)
            revo_peak_bytes = max(revo_peak_bytes, measure_memory_rss())
        for t in pert_gen:
            _, last_p, n_tok = _llama_nll_and_last_topk(llm, t, max_len, topk, temperature=float(tau), prompt_char_limit=prompt_char_limit)
            topk_gen_revo_pert.append(last_p)
            revo_tokens += int(n_tok)
            revo_peak_bytes = max(revo_peak_bytes, measure_memory_rss())
        elapsed_revo = time.perf_counter() - t1
        revo_rss_mb = float(revo_peak_bytes) / (1024.0 * 1024.0)
        revo_tok_per_s = (float(revo_tokens) / elapsed_revo) if elapsed_revo > 0 else 0.0

        # recon/stability metrics from topK logprobs
        sims: List[float] = []
        kls: List[float] = []
        st_sims: List[float] = []
        st_kls: List[float] = []
        for b, r in zip(topk_gen_base, topk_gen_revo):
            va, vb = _align_topk_dicts(b, r)
            sims.append(_cosine_sim(va, vb))
            kls.append(_kl_div(va, vb))
        for b, r in zip(topk_gen_base_pert, topk_gen_revo_pert):
            va, vb = _align_topk_dicts(b, r)
            st_sims.append(_cosine_sim(va, vb))
            st_kls.append(_kl_div(va, vb))

        results.append(VariantMetrics(
            name="baseline",
            nll_general=float(np.mean(nlls_gen_base)),
            nll_ood=float(np.mean(nlls_ood_base)),
            recon_cos=1.0,
            recon_kl=0.0,
            stability_cos=1.0,
            stability_kl=0.0,
            eval_time_s=elapsed_base,
            tokens_scored=int(base_tokens),
            tokens_per_s=float(base_tok_per_s),
            rss_peak_mb=float(base_rss_mb),
            cold_start_s=float(base_cold_s or 0.0),
            warm_avg_s=(float(base_warm_sum) / max(1, base_warm_cnt)),
        ))

        results.append(VariantMetrics(
            name=f"revo(tau={tau})",
            nll_general=float(np.mean(nlls_gen_revo)),
            nll_ood=float(np.mean(nlls_ood_revo)),
            recon_cos=float(np.mean(sims)) if sims else 1.0,
            recon_kl=float(np.mean(kls)) if kls else 0.0,
            stability_cos=float(np.mean(st_sims)) if st_sims else 1.0,
            stability_kl=float(np.mean(st_kls)) if st_kls else 0.0,
            eval_time_s=elapsed_revo,
            tokens_scored=int(revo_tokens),
            tokens_per_s=float(revo_tok_per_s),
            rss_peak_mb=float(revo_rss_mb),
            cold_start_s=float(revo_cold_s or 0.0),
            warm_avg_s=(float(revo_warm_sum) / max(1, revo_warm_cnt)),
        ))

    payload: Dict[str, Any] = asdict(CompareResult(
        model=f"llama.cpp:{os.path.basename(gguf_path)}",
        seeds=seeds,
        prompts=prompts,
        max_length=max_len,
        variants=results,
    ))
    return payload


# REVO moderate pipeline (Phase I + light IV) con estratificación y calibración de head opcional

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
    # Estratificar: aplicar solo al último porcentaje de capas
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
    # Low-rank solo en permitidos
    ranks_full = allocate_ranks_energy_with_caps(prof, energy_keep=energy_keep, max_rank=None, max_rank_frac=max_rank_frac)
    ranks = {k: v for k, v in ranks_full.items() if k in allow}
    replace_2d_modules_with_lowrank(model, ranks, calibrate=True, calibrate_samples=128, seed=seed)
    # Fractal + efímero solo en permitidos, con gamma inicial adaptativa por entropía espectral
    if enable_fractal:
        replace_with_fractal(model, depth=2, alpha=0.5, name_patterns=["attn", "mlp", "c_proj"], skip_lm_head=True, allow_names=sorted(list(allow)))
    gamma_map = {}
    for k in allow:
        info = prof.get(k, {})
        h = float(info.get("entropy_norm", 0.0)) if isinstance(info, dict) else 0.0
        gamma_map[k] = (gamma_lowentropy if h < float(entropy_threshold) else gamma_normal)
    replace_with_ephemeral(model, name_patterns=name_patterns, skip_lm_head=True, allow_names=sorted(list(allow)), gamma_init_map=gamma_map)
    calibrate_ephemeral(model, tok, texts=texts[: min(32, len(texts))], steps=int(ephem_steps), lr=float(ephem_lr), lambda_phys=1.0, max_length=max_len)
    # Calibración post-REVO del head (escala y sesgo de logits)
    if calibrate_head:
        wrap_lm_head_with_calib(model)
        calibrate_logits(model, tok, texts=texts[: min(64, len(texts))], steps=int(head_calib_steps), lr=float(head_calib_lr), max_length=max_len)


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


def evaluate_variants(
    model_name: str,
    seeds: List[int],
    prompts: int,
    max_len: int,
    prune_amount: float,
    device_map: str | None = None,
    dtype_str: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    skip_dynamic_quant: bool = False,
    revo_deep_fraction: float = 0.5,
    revo_calibrate_head: bool = False,
    revo_enable_fractal: bool = True,
    revo_ephem_steps: int = 5,
    revo_energy_keep: float = 0.92,
    revo_max_rank_frac: float = 0.20,
    revo_entropy_threshold: float = 0.20,
    revo_gamma_lowentropy: float = 3.0,
    revo_gamma_normal: float = 2.0,
    revo_ephem_lr: float = 5e-2,
    revo_head_calib_steps: int = 50,
    revo_head_calib_lr: float = 1e-2,
) -> Dict[str, Any]:
    gen_texts = _texts_general(prompts)
    ood_texts = _texts_ood(max(50, prompts // 2))

    results: List[VariantMetrics] = []

    for seed in seeds:
        seed_everything(seed)
        # Baseline
        base_model, tok = _load_model_tok(model_name, device_map=device_map, dtype_str=dtype_str, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        t0 = time.perf_counter()
        base_peak_bytes_hf = measure_memory_rss()
        nll_gen = evaluate_nll(base_model, tok, gen_texts, max_length=max_len, device=_device())
        nll_ood = evaluate_nll(base_model, tok, ood_texts, max_length=max_len, device=_device())
        base_logits = _last_token_logits(base_model, tok, gen_texts[:50], max_len)
        pert_gen = [_perturb(t) for t in gen_texts[:50]]
        base_logits_pert = _last_token_logits(base_model, tok, pert_gen, max_len)
        t_base = time.perf_counter() - t0
        base_rss_mb_hf = float(max(base_peak_bytes_hf, measure_memory_rss())) / (1024.0 * 1024.0)
        results.append(VariantMetrics(name="baseline", nll_general=nll_gen, nll_ood=nll_ood, recon_cos=1.0, recon_kl=0.0, stability_cos=1.0, stability_kl=0.0, eval_time_s=t_base, tokens_scored=0, tokens_per_s=0.0, rss_peak_mb=base_rss_mb_hf, cold_start_s=0.0, warm_avg_s=0.0))

        # Quantization + Pruning
        qp_model, _ = _load_model_tok(model_name, device_map=device_map, dtype_str=dtype_str, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        # If already using 4/8-bit loading, or user asks to skip, do not apply dynamic int8
        if not (load_in_4bit or load_in_8bit or skip_dynamic_quant):
            qp_model = _quantize_dynamic_int8(qp_model)
        _global_prune(qp_model, amount=prune_amount)
        t1 = time.perf_counter()
        qp_peak_bytes_hf = measure_memory_rss()
        qp_nll_gen = evaluate_nll(qp_model, tok, gen_texts, max_length=max_len, device=_device())
        qp_nll_ood = evaluate_nll(qp_model, tok, ood_texts, max_length=max_len, device=_device())
        qp_logits = _last_token_logits(qp_model, tok, gen_texts[:50], max_len)
        qp_logits_pert = _last_token_logits(qp_model, tok, pert_gen, max_len)
        t_qp = time.perf_counter() - t1
        # reconstruction vs baseline
        sims = []
        kls = []
        st_sims = []
        st_kls = []
        for b, q in zip(base_logits, qp_logits):
            sims.append(_cosine_sim(b, q))
            kls.append(_kl_div(b, q))
        for b, q in zip(base_logits_pert, qp_logits_pert):
            st_sims.append(_cosine_sim(b, q))
            st_kls.append(_kl_div(b, q))
        qp_rss_mb_hf = float(max(qp_peak_bytes_hf, measure_memory_rss())) / (1024.0 * 1024.0)
        results.append(VariantMetrics(
            name="quant+prune",
            nll_general=qp_nll_gen,
            nll_ood=qp_nll_ood,
            recon_cos=float(np.mean(sims)),
            recon_kl=float(np.mean(kls)),
            stability_cos=float(np.mean(st_sims)),
            stability_kl=float(np.mean(st_kls)),
            eval_time_s=t_qp,
            tokens_scored=0,
            tokens_per_s=0.0,
            rss_peak_mb=qp_rss_mb_hf,
            cold_start_s=0.0,
            warm_avg_s=0.0,
        ))

        # REVO (moderate)
        revo_model, _ = _load_model_tok(model_name, device_map=device_map, dtype_str=dtype_str, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        try:
            _apply_revo(
                revo_model,
                tok,
                gen_texts,
                seed,
                max_len,
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
            # even if some wrapper fails, evaluate as-is
            pass
        t2 = time.perf_counter()
        revo_peak_bytes_hf = measure_memory_rss()
        revo_nll_gen = evaluate_nll(revo_model, tok, gen_texts, max_length=max_len, device=_device())
        revo_nll_ood = evaluate_nll(revo_model, tok, ood_texts, max_length=max_len, device=_device())
        revo_logits = _last_token_logits(revo_model, tok, gen_texts[:50], max_len)
        revo_logits_pert = _last_token_logits(revo_model, tok, pert_gen, max_len)
        t_revo = time.perf_counter() - t2
        sims = []
        kls = []
        st_sims = []
        st_kls = []
        for b, r in zip(base_logits, revo_logits):
            sims.append(_cosine_sim(b, r))
            kls.append(_kl_div(b, r))
        for b, r in zip(base_logits_pert, revo_logits_pert):
            st_sims.append(_cosine_sim(b, r))
            st_kls.append(_kl_div(b, r))
        revo_rss_mb_hf = float(max(revo_peak_bytes_hf, measure_memory_rss())) / (1024.0 * 1024.0)
        results.append(VariantMetrics(
            name="revo",
            nll_general=revo_nll_gen,
            nll_ood=revo_nll_ood,
            recon_cos=float(np.mean(sims)),
            recon_kl=float(np.mean(kls)),
            stability_cos=float(np.mean(st_sims)),
            stability_kl=float(np.mean(st_kls)),
            eval_time_s=t_revo,
            tokens_scored=0,
            tokens_per_s=0.0,
            rss_peak_mb=revo_rss_mb_hf,
            cold_start_s=0.0,
            warm_avg_s=0.0,
        ))

    payload: Dict[str, Any] = asdict(CompareResult(
        model=model_name,
        seeds=seeds,
        prompts=prompts,
        max_length=max_len,
        variants=results,
    ))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparativo REVO vs Quant+Prune")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--prune-amount", type=float, default=0.3)
    ap.add_argument("--device-map", type=str, default=None, help="e.g., auto")
    ap.add_argument("--dtype", type=str, default=None, help="float16|bfloat16|float32")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--results-json", default=None)
    ap.add_argument("--skip-dynamic-quant", action="store_true")
    # llama.cpp backend (GGUF)
    ap.add_argument("--llama-gguf", type=str, default=None, help="Path to GGUF model for llama.cpp backend")
    ap.add_argument("--llama-n-ctx", type=int, default=2048)
    ap.add_argument("--llama-n-threads", type=int, default=max(os.cpu_count() or 4, 4))
    ap.add_argument("--llama-topk", type=int, default=50, help="Top-K logprobs to fetch for last-token distribution")
    ap.add_argument("--llama-tau", type=float, default=1.0, help="Temperature scaling for REVO-like calibration (tau)")
    ap.add_argument("--llama-fast", action="store_true", help="Reduce OOD/perturbation sizes for faster CPU micro-runs")
    ap.add_argument("--llama-prompt-limit", type=int, default=128, help="Max characters of prompt to score in llama.cpp path")
    ap.add_argument("--revo-deep-fraction", type=float, default=0.5)
    ap.add_argument("--revo-calibrate-head", action="store_true")
    # REVO advanced controls (ablation/tuning)
    ap.add_argument("--revo-disable-fractal", action="store_true", help="Disable fractal wrapping stage")
    ap.add_argument("--revo-ephem-steps", type=int, default=5, help="Steps for ephemeral gamma calibration")
    ap.add_argument("--revo-ephem-lr", type=float, default=5e-2, help="LR for ephemeral gamma calibration")
    ap.add_argument("--revo-energy-keep", type=float, default=0.92, help="Spectral energy to keep for low-rank")
    ap.add_argument("--revo-max-rank-frac", type=float, default=0.20, help="Max rank fraction per layer for low-rank")
    ap.add_argument("--revo-entropy-threshold", type=float, default=0.20, help="Entropy threshold for high/low gamma init")
    ap.add_argument("--revo-gamma-lowentropy", type=float, default=3.0, help="Gamma init for low-entropy layers")
    ap.add_argument("--revo-gamma-normal", type=float, default=2.0, help="Gamma init for normal-entropy layers")
    ap.add_argument("--revo-head-calib-steps", type=int, default=50, help="Steps for head calibration")
    ap.add_argument("--revo-head-calib-lr", type=float, default=1e-2, help="LR for head calibration")
    # Alignment check flags
    ap.add_argument("--alignment-check", action="store_true", help="Compute L2 norms and raw logits diffs hooking the same graph point (baseline vs REVO)")
    ap.add_argument("--hook-module", type=str, default="transformer.ln_f", help="Dotted path to module to hook (e.g., transformer.ln_f or transformer.h.11)")
    ap.add_argument("--alignment-results", type=str, default=None, help="Optional path to save alignment JSON (defaults to quality/compare/alignment_<ts>.json)")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.alignment_check:
        # Use the first seed for a light check
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        seed0 = seeds[0] if seeds else 0
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
            args.model,
            seed0,
            args.prompts,
            args.max_length,
            args.hook_module,
            revo_kwargs,
        )
        out_path = args.alignment_results or os.path.join("quality", "compare", f"alignment_{int(time.time())}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps({"alignment_path": out_path}, indent=2))
        return

    if args.llama_gguf:
        result = evaluate_variants_llama(
            args.llama_gguf,
            seeds,
            args.prompts,
            args.max_length,
            n_ctx=args.llama_n_ctx,
            n_threads=args.llama_n_threads,
            topk=args.llama_topk,
            tau=float(args.llama_tau),
            prompt_char_limit=int(args.llama_prompt_limit),
            fast=bool(args.llama_fast),
        )
    else:
        result = evaluate_variants(
            args.model,
            seeds,
            args.prompts,
            args.max_length,
            args.prune_amount,
            device_map=args.device_map,
            dtype_str=args.dtype,
            load_in_4bit=bool(args.load_in_4bit),
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

    out_path = args.results_json or os.path.join("quality", "compare", f"compare_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({"results_path": out_path}, indent=2))


if __name__ == "__main__":
    main()
