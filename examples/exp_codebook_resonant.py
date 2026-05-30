"""ResonantCodebook — benchmark end-to-end.

Compara: HyperLoRA baseline vs Codebook puro vs Codebook seeded.
Mide: NLL delta, fraction improved, codebook growth, consolidation.
"""

from __future__ import annotations

import json
import time
import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revo._logging import get_logger
from revo.ephemeral_engine import EphemeralConfig, EphemeralEngine

log = get_logger(__name__)

LONG_TEXTS = [
    "The capital of France is Paris and it has been for centuries a center of art and culture",
    "Quantum computing relies on superposition and entanglement to solve problems that classical computers cannot",
    "In the beginning there was nothing and then the universe expanded from a singularity",
    "The theory of relativity shows that space and time are curved by mass and energy",
    "Machine learning models can recognize patterns in data and make predictions about future events",
    "The history of Rome began as a small settlement on the Tiber River and grew into an empire",
    "Neural networks are composed of layers of interconnected nodes that process information",
    "The speed of light in vacuum is approximately three hundred million meters per second",
    "In mathematics a prime number is a natural number greater than one that cannot be formed by multiplying two smaller",
    "The human brain processes information through a network of billions of neurons connected by synapses",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose",
    "The Industrial Revolution transformed society by introducing machines that could produce goods faster",
    "DNA contains the genetic instructions for the development and functioning of all known living organisms",
    "The Renaissance was a period of cultural artistic and scientific rebirth in Europe",
    "Climate change refers to long term shifts in temperature and weather patterns",
]


@torch.no_grad()
def run_benchmark(model, tokenizer, texts, cfg: EphemeralConfig, label: str = ""):
    device = next(model.parameters()).device
    lm_head = model.lm_head
    W_ref = lm_head.weight.detach().float().cpu().numpy().copy()

    engine = EphemeralEngine(model, cfg)
    total_base, total_mod = 0.0, 0.0
    all_deltas, all_log = [], []
    n_tokens, n_skipped = 0, 0
    t_start = time.perf_counter()

    for text in texts:
        engine.reset_sequence()
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        ids = inp.input_ids.to(device)
        slen = ids.shape[1]
        if slen < 2:
            continue

        logits_base = model(ids).logits
        nll_pertok = torch.nn.functional.cross_entropy(
            logits_base[0, :-1].float(), ids[0, 1:], reduction="none"
        ).cpu().numpy()

        hs = model.transformer.wte(ids)
        hs = hs + model.transformer.wpe(torch.arange(slen, device=device))
        for block in model.transformer.h:
            hs = block(hs)[0]
        hs = model.transformer.ln_f(hs)

        for pos in range(slen - 1):
            h_pos = hs[0, pos]
            nll_base = float(nll_pertok[pos])
            logits_mod, meta = engine.step(h_pos, pos)

            if logits_mod is None:
                n_skipped += 1
                total_base += nll_base
                all_deltas.append(0.0)
                all_log.append({**meta, "nll_delta": 0.0, "nll_base": nll_base})
                continue

            logits_at = logits_mod
            target = ids[0, pos + 1]
            if logits_at.dim() == 1:
                logits_at = logits_at.unsqueeze(0)
            nll_mod = torch.nn.functional.cross_entropy(
                logits_at.float(), target.unsqueeze(0)
            ).item()
            nll_delta = nll_mod - nll_base

            engine.record_nll_delta(nll_delta)
            total_base += nll_base
            total_mod += nll_mod
            all_deltas.append(nll_delta)
            all_log.append({**meta, "nll_delta": nll_delta, "nll_base": nll_base,
                            "nll_mod": nll_mod})
            n_tokens += 1

    total_time = time.perf_counter() - t_start

    W_final = lm_head.weight.detach().float().cpu().numpy()
    drift = float(np.max(np.abs(W_final - W_ref)))
    drift_ok = drift < 1e-5

    estats = engine.stats()

    result = {
        "config": asdict(cfg),
        "label": label,
        "n_tokens": n_tokens,
        "n_skipped": n_skipped,
        "total_n_tokens": len(all_deltas),
        "baseline_nll": total_base,
        "modified_nll": total_mod,
        "nll_delta": total_mod - total_base,
        "nll_delta_per_token": (total_mod - total_base) / max(1, n_tokens),
        "fraction_improved": sum(1 for d in all_deltas if d < -1e-10) / max(1, len(all_deltas)),
        "fraction_degraded": sum(1 for d in all_deltas if d > 1e-10) / max(1, len(all_deltas)),
        "mean_delta": float(np.mean(all_deltas)) if all_deltas else 0.0,
        "std_delta": float(np.std(all_deltas)) if len(all_deltas) > 1 else 0.0,
        "reversibility_drift": drift,
        "reversibility_ok": drift_ok,
        "total_time_s": total_time,
        "time_per_token_ms": total_time / max(1, n_tokens) * 1000,
        "engine_stats": estats,
    }
    return result


def summary(r: dict) -> str:
    lines = [
        f"\n{'='*65}",
        f"  {r['label']}",
        f"  {'='*61}",
    ]
    if r.get('reversibility_ok', False):
        lines.append(f"  ✓ drift={r['reversibility_drift']:.2e}")
    else:
        lines.append(f"  ✗ REVERSIBILIDAD FALLÓ: drift={r['reversibility_drift']:.2e}")

    lines.extend([
        f"\n  NLL delta:          {r['nll_delta']:+8.4f}  ({r['nll_delta_per_token']:+8.6f}/tok)",
        f"  Std(Δ):             {r['std_delta']:.4f}",
        f"  Improved:           {r['fraction_improved']:.1%}  Degraded: {r['fraction_degraded']:.1%}",
        f"\n  Tokens procesados:  {r['n_tokens']} (+{r['n_skipped']} skipped)",
        f"  Time:               {r['total_time_s']:.1f}s ({r['time_per_token_ms']:.0f}ms/tok)",
    ])
    es = r['engine_stats']
    lines.extend([
        f"\n  ENGINE STATS:",
        f"    Cache hits (exact):    {es['cache_hits']}",
        f"    Cache misses:          {es['cache_misses']}",
        f"    Gated skip:            {es['gated_skip']}",
    ])
    if "codebook" in es:
        cb = es["codebook"]
        lines.extend([
            f"    Codebook size:         {cb['codebook_size']}/{cb['max_k']}",
            f"    Codebook contribs:     {es['codebook_contribs']}",
            f"    Codebook additions:    {es['codebook_additions']}",
            f"    Codebook mean fitness: {cb['mean_fitness']:.4f}",
            f"    Codebook delta cache:  {cb['delta_cache_size']}",
        ])
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ResonantCodebook Benchmark")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompts", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-json", default="quality/codebook/benchmark.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    texts = LONG_TEXTS[:args.prompts]

    print(f"Model: {args.model}, {args.prompts} prompts, seed={args.seed}")

    # === 1. BASELINE: HyperLoRA ===
    cfg_hl = EphemeralConfig(
        use_codebook=False, use_cache=True, rank=4, window_size=4,
        novelty_threshold=99.0,  # disable novelty gate for fair comparison
        delta_window_threshold=99.0,
        scale_max=0.8,
    )
    r1 = run_benchmark(model, tokenizer, texts, cfg_hl, label="BASELINE: HyperLoRA")
    print(summary(r1))

    # === 2. CODEBOOK PURO (random init, sin seeding) ===
    cfg_cb = EphemeralConfig(
        use_codebook=True, use_cache=False, rank=4, window_size=4,
        novelty_threshold=99.0, delta_window_threshold=99.0, scale_max=0.8,
        codebook_k=64, codebook_max_k=256, codebook_lr=0.15,
        codebook_top_k=3, codebook_exploration=0.04,
        codebook_add_threshold=0.7, codebook_seed_from_hyperlora=0,
    )
    r2 = run_benchmark(model, tokenizer, texts, cfg_cb, label="CODEBOOK: puro (random init)")
    print(summary(r2))

    # === 3. CODEBOOK SEEDED (warmup con HyperLoRA) ===
    cfg_cb_s = EphemeralConfig(
        use_codebook=True, use_cache=False, rank=4, window_size=4,
        novelty_threshold=99.0, delta_window_threshold=99.0, scale_max=0.8,
        codebook_k=64, codebook_max_k=256, codebook_lr=0.15,
        codebook_top_k=3, codebook_exploration=0.04,
        codebook_add_threshold=0.7, codebook_seed_from_hyperlora=64,
    )
    r3 = run_benchmark(model, tokenizer, texts, cfg_cb_s, label="CODEBOOK: seeded (64 warmup HL)")
    print(summary(r3))

    combined = {
        "baseline_hyperlora": r1,
        "codebook_puro": r2,
        "codebook_seeded": r3,
    }
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info("Saved to %s", args.results_json)
