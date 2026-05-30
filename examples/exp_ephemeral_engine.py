"""REVO EphemeralEngine — test end-to-end completo.

Ejercita: context encoder, selective gating, ModeCache con similitud Z.
Mide: NLL delta, reversibilidad, cache hits, gate statistics.
"""

from __future__ import annotations

import json
import time
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataclasses import asdict
from revo._logging import get_logger
from revo.context_encoder import WindowContextEncoder
from revo.ephemeral_engine import EphemeralConfig, EphemeralEngine
from revo.engine import apply_delta, revert_delta

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
def run_full_test(model, tokenizer, texts, cfg: EphemeralConfig, label: str = ""):
    device = next(model.parameters()).device
    lm_head = model.lm_head
    W_ref = lm_head.weight.detach().float().cpu().numpy().copy()

    engine = EphemeralEngine(model, cfg)
    total_base, total_mod = 0.0, 0.0
    all_deltas, all_log = [], []
    n_tokens, n_skipped = 0, 0
    token_times = []
    t_start = time.perf_counter()

    for text in texts:
        engine.reset()
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        ids = inp.input_ids.to(device)
        slen = ids.shape[1]
        if slen < 2:
            continue

        # Baseline: forward with original head
        logits_base = model(ids).logits
        nll_pertok = torch.nn.functional.cross_entropy(
            logits_base[0, :-1].float(), ids[0, 1:], reduction="none"
        ).cpu().numpy()

        # Get hidden states for all positions (one forward)
        hs = model.transformer.wte(ids)
        hs = hs + model.transformer.wpe(torch.arange(slen, device=device))
        for block in model.transformer.h:
            hs = block(hs)[0]
        hs = model.transformer.ln_f(hs)

        for pos in range(slen - 1):
            h_pos = hs[0, pos]
            nll_base = float(nll_pertok[pos])

            # Ephemeral step
            logits_mod, meta = engine.step(h_pos, pos)

            if logits_mod is None:
                n_skipped += 1
                total_base += nll_base
                all_deltas.append(0.0)
                all_log.append({**meta, "nll_delta": 0.0, "nll_base": nll_base})
                continue

            # Compute NLL
            logits_at = logits_mod  # (hidden,) or (1, hidden)
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

    # Reversibility check
    W_final = lm_head.weight.detach().float().cpu().numpy()
    drift = float(np.max(np.abs(W_final - W_ref)))
    assert drift < 1e-5, f"Reversibility failed: drift={drift:.2e}"
    drift_ok = drift < 1e-6

    # Engine stats
    estats = engine.stats()

    # Aggregate
    deltas_no_skip = [d for d in all_deltas if abs(d) > 1e-10]
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
        "fraction_neutral": sum(1 for d in all_deltas if abs(d) <= 1e-10) / max(1, len(all_deltas)),
        "fraction_degraded": sum(1 for d in all_deltas if d > 1e-10) / max(1, len(all_deltas)),
        "tokens_degraded_gt_0_1": sum(1 for d in all_deltas if d > 0.1),
        "tokens_improved_gt_0_1": sum(1 for d in all_deltas if d < -0.1),
        "mean_delta": float(np.mean(all_deltas)) if all_deltas else 0.0,
        "std_delta": float(np.std(all_deltas)) if len(all_deltas) > 1 else 0.0,
        "max_delta": float(np.max(all_deltas)),
        "min_delta": float(np.min(all_deltas)),
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
        lines.append(f"  ✓ Reversibilidad: drift={r['reversibility_drift']:.2e}")
    else:
        lines.append(f"  ✗ REVERSIBILIDAD FALLÓ: drift={r['reversibility_drift']:.2e}")

    lines.extend([
        f"\n  NLL delta:          {r['nll_delta']:+8.4f}  ({r['nll_delta_per_token']:+8.6f}/tok)",
        f"  Std(Δ):             {r['std_delta']:.4f}",
        f"  Range:              [{r['min_delta']:+8.4f}, {r['max_delta']:+8.4f}]",
        f"  Degraded >0.1:      {r['tokens_degraded_gt_0_1']}/{r['total_n_tokens']}",
        f"  Improved >0.1:      {r['tokens_improved_gt_0_1']}/{r['total_n_tokens']}",
        f"  Improved:           {r['fraction_improved']:.1%}  Neutral: {r['fraction_neutral']:.1%}  Degraded: {r['fraction_degraded']:.1%}",
        f"\n  Tokens procesados:  {r['n_tokens']} (+{r['n_skipped']} skipped por gating)",
        f"  Time:               {r['total_time_s']:.1f}s ({r['time_per_token_ms']:.0f}ms/tok)",
        f"\n  ENGINE STATS:",
    ])
    es = r['engine_stats']
    lines.extend([
        f"    Cache hits (exact):    {es['cache_hits']}",
        f"    Cache hits (similar):  {es['cache_similar_hits']}",
        f"    Cache misses:          {es['cache_misses']}",
        f"    Gated skip:            {es['gated_skip']}",
        f"    Gated reduced:         {es['gated_reduced']}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="REVO EphemeralEngine Test")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-json", default="quality/ephemeral/engine_test.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    texts = LONG_TEXTS[:args.prompts]

    print(f"Modelo: {args.model}, {args.prompts} prompts, seed={args.seed}")

    # Test 1: engine COMPLETO (context encoder + gating + cache)
    cfg1 = EphemeralConfig(use_cache=True, rank=4, window_size=4,
                            novelty_threshold=0.3, delta_window=3,
                            scale_max=0.8)
    r1 = run_full_test(model, tokenizer, texts, cfg1, label="FULL ENGINE (window4 + gate + cache)")
    print(summary(r1))

    # Test 2: same engine WITHOUT gating (to measure gate impact)
    cfg2 = EphemeralConfig(use_cache=True, rank=4, window_size=4,
                            novelty_threshold=99.0,  # effectively off
                            delta_window_threshold=99.0, scale_max=0.8)
    r2 = run_full_test(model, tokenizer, texts, cfg2, label="ENGINE NO GATE (window4 + cache only)")
    print(summary(r2))

    # Test 3: full engine with SECOND pass (to measure cache hits from similar contexts)
    # Re-use same config, which should have populated cache from test 1
    # But we need to test on different (but similar) prompts to measure similarity cache
    similar_texts = [
        "The capital of France is Paris the city of light and love",
        "Quantum computing uses qubits and superposition to solve hard problems",
        "In the beginning there was the Big Bang and then everything expanded",
    ]
    log.info("Test 3: Similar prompts for cache hits...")
    r3 = run_full_test(model, tokenizer, similar_texts, cfg1, label="CACHE TEST (similar prompts)")
    print(summary(r3))

    # Combined output
    combined = {"full_engine": r1, "no_gate": r2, "cache_test": r3}
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info("Saved to %s", args.results_json)
