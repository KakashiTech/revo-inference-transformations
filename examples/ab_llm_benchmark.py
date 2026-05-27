from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo.features import build_features, build_context_vector, compute_mode_key
from revo.hyperlora import HyperLoraConfig, hyperlora_generate
from revo.torch_integration import apply_delta_torch, revert_delta_torch


def _pick_linear_module(model: torch.nn.Module) -> torch.nn.Linear:
    # Prefer last block's feed-forward input projection if available (GPT2-like)
    try:
        return model.transformer.h[-1].mlp.c_fc  # type: ignore[attr-defined]
    except Exception:
        pass
    # Fallback: first Linear with 2D weight
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.weight.dim() == 2:
            return m
    raise RuntimeError("No suitable Linear module found for delta application")


@dataclass
class GenMetrics:
    latency_s: float
    rss_before: int
    rss_after: int
    mem_delta: int
    num_new_tokens: int
    tokens_per_sec: float
    words_total: int
    vocab_size: int
    lex_entropy_norm: float
    bigram_cov: float
    avg_sent_len: float
    list_ratio: float


def _quality_proxy(text: str) -> Tuple[int, int, float, float, float, float]:
    # Mirrors the heuristics used in observability.log_identity_anchor
    txt = (text or "")
    tmp = txt.replace("?", ".").replace("!", ".")
    sents = [s.strip() for s in tmp.split(".") if s.strip()]
    words = [w.strip().lower() for w in txt.split() if w.strip()]
    V = len(set(words))
    N = max(1, len(words))
    # Lexical entropy normalized by log(V)
    from collections import Counter
    import math

    freqs = Counter(words)
    ent = 0.0
    for c in freqs.values():
        p = float(c) / float(N)
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    ent_norm = float(ent / max(1e-9, math.log(float(max(1, V)) + 1e-9))) if V > 1 else 0.0

    bigrams = [(words[i], words[i + 1]) for i in range(0, max(0, len(words) - 1))]
    B = len(bigrams)
    Ub = len(set(bigrams))
    bigram_cov = float(Ub) / float(max(1, B))

    lens = [len(s.split()) for s in sents] or [0]
    avg_len = float(sum(lens)) / float(len(lens))

    lines = [ln.strip().lower() for ln in txt.splitlines()]
    list_like = sum(1 for ln in lines if ln.startswith(("- ", "* ", "1.", "2.", "3.")))
    list_ratio = float(list_like) / float(max(1, len(lines)))

    return N, V, ent_norm, bigram_cov, avg_len, list_ratio


def _measure_mem() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


def _gen_once(model, tok, prompt: str, max_new_tokens: int, do_sample: bool) -> Tuple[str, int]:
    tokens = tok(prompt, return_tensors="pt")
    in_len = int(tokens.input_ids.shape[-1])
    with torch.no_grad():
        out = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    out_len = int(out.shape[-1])
    return text, int(max(0, out_len - in_len))


def run_trial(model, tok, prompt: str, cfg: HyperLoraConfig, seed: int, apply_revo: bool, max_new_tokens: int, do_sample: bool) -> GenMetrics:
    torch.manual_seed(seed)
    np.random.seed(seed)

    target = _pick_linear_module(model)
    of, inf = target.weight.shape

    handle = None
    if apply_revo:
        feats = build_features(prompt)
        mode_key = compute_mode_key(feats, prompt)
        ctx = build_context_vector(feats, context_dim=cfg.context_dim, seed=seed)
        A, B, scale = hyperlora_generate(ctx, cfg)
        # Save copy to verify reversibility
        W0 = target.weight.detach().clone()
        handle = apply_delta_torch(target, A, B, scale)

    rss_before = _measure_mem()
    t0 = time.perf_counter()
    text, new_tokens = _gen_once(model, tok, prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
    latency = time.perf_counter() - t0
    rss_after = _measure_mem()

    if handle is not None:
        revert_delta_torch(handle)
        # Reversibility check
        assert torch.allclose(target.weight, W0, atol=1e-6), "Reversibility check failed (torch)"

    N, V, ent_norm, bigram_cov, avg_len, list_ratio = _quality_proxy(text)
    tps = (float(new_tokens) / latency) if latency > 0 and new_tokens > 0 else 0.0
    return GenMetrics(
        latency_s=float(latency),
        rss_before=int(rss_before),
        rss_after=int(rss_after),
        mem_delta=int(rss_after - rss_before),
        num_new_tokens=int(new_tokens),
        tokens_per_sec=float(tps),
        words_total=int(N),
        vocab_size=int(V),
        lex_entropy_norm=float(ent_norm),
        bigram_cov=float(bigram_cov),
        avg_sent_len=float(avg_len),
        list_ratio=float(list_ratio),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="A/B benchmark for REVO deltas on a tiny LLM (CPU)")
    p.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    p.add_argument("--seed", type=int, default=int(os.environ.get("REVO_SEED", "0") or 0))
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--rank", type=int, default=int(os.environ.get("REVO_RANK", "8") or 8))
    p.add_argument("--ctx-dim", type=int, default=int(os.environ.get("REVO_CTX_DIM", "64") or 64))
    p.add_argument("--hidden-dim", type=int, default=int(os.environ.get("REVO_HIDDEN_DIM", "128") or 128))
    p.add_argument("--prompts", nargs="*", default=None, help="List of prompts; if empty, uses defaults")
    p.add_argument("--results-jsonl", default=None, help="Optional path to write per-trial JSONL records")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    prompts = args.prompts or [
        "Hola REVO! Dame dos bullets sobre deltas low-rank.",
        "Explain ephemeral low-rank adapters in two lines.",
        "Resume en 1 frase qué es un mode cache.",
        "List three pros of reversible parameter deltas.",
        "¿Por qué acotar la escala de un delta es útil?",
    ]

    cfg = HyperLoraConfig(context_dim=args.ctx_dim, rank=args.rank, in_features=0, out_features=0, hidden_dim=args.hidden_dim)

    # Prepare results file
    results_path: Optional[str] = args.results_jsonl
    if results_path is None:
        ts = int(time.time())
        results_path = os.path.join("quality", "ab_runs", f"ab_{ts}.jsonl")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    agg: Dict[str, List[float]] = {k: [] for k in [
        "latency_base", "latency_revo", "mem_delta_base", "mem_delta_revo",
        "tps_base", "tps_revo", "lex_base", "lex_revo"
    ]}

    with open(results_path, "a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            # For each prompt, re-derive cfg in/out dims from the chosen module
            target = _pick_linear_module(model)
            of, inf = target.weight.shape
            cfg.in_features = int(inf)
            cfg.out_features = int(of)

            base = run_trial(
                model, tok, prompt, cfg, seed=args.seed, apply_revo=False,
                max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
            )
            revo = run_trial(
                model, tok, prompt, cfg, seed=args.seed, apply_revo=True,
                max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
            )

            rec = {
                "idx": i,
                "prompt": prompt,
                "baseline": asdict(base),
                "revo": asdict(revo),
                "deltas": {
                    "latency_diff_s": revo.latency_s - base.latency_s,
                    "mem_delta_diff": revo.mem_delta - base.mem_delta,
                    "tps_diff": revo.tokens_per_sec - base.tokens_per_sec,
                    "lex_entropy_diff": revo.lex_entropy_norm - base.lex_entropy_norm,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            agg["latency_base"].append(base.latency_s)
            agg["latency_revo"].append(revo.latency_s)
            agg["mem_delta_base"].append(base.mem_delta)
            agg["mem_delta_revo"].append(revo.mem_delta)
            agg["tps_base"].append(base.tokens_per_sec)
            agg["tps_revo"].append(revo.tokens_per_sec)
            agg["lex_base"].append(base.lex_entropy_norm)
            agg["lex_revo"].append(revo.lex_entropy_norm)

    def _mean(x: List[float]) -> float:
        return float(sum(x) / max(1, len(x)))

    summary = {
        "n": len(prompts),
        "latency_base_mean_s": _mean(agg["latency_base"]),
        "latency_revo_mean_s": _mean(agg["latency_revo"]),
        "mem_delta_base_mean": _mean(agg["mem_delta_base"]),
        "mem_delta_revo_mean": _mean(agg["mem_delta_revo"]),
        "tps_base_mean": _mean(agg["tps_base"]),
        "tps_revo_mean": _mean(agg["tps_revo"]),
        "lex_base_mean": _mean(agg["lex_base"]),
        "lex_revo_mean": _mean(agg["lex_revo"]),
        "results_path": results_path,
    }

    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
