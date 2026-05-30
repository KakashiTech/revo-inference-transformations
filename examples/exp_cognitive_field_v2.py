"""REVO proto-campo cognitivo V2 — field as context for HyperLoRA.

The cognitive field (Φ/A/C) does NOT override scale after generation.
Instead, field signals are INJECTED into the context vector before
HyperLoRA generation. This makes the field part of the generative law:

  ctx' = [ctx | entropy | coherence | prev_nll_delta]
  A, B, scale = HyperLoRA(ctx')

This is architecturally cleaner: the generation law sees the field state
and can learn to respond to it deterministically.
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._logging import get_logger
from revo.engine import apply_delta, revert_delta
from revo.hyperlora import HyperLoraConfig, hyperlora_generate

log = get_logger(__name__)


@dataclass
class FieldConfig:
    model_name: str = "gpt2"
    rank: int = 4
    base_context_dim: int = 16
    field_dim: int = 3
    max_length: int = 32
    prompts: int = 8
    seed: int = 0


def compute_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    return float(-torch.sum(probs * torch.log(probs.clamp(min=1e-10)), dim=-1).mean().item())


def compute_coherence(current_hs: np.ndarray, history: List[np.ndarray]) -> float:
    if not history:
        return 0.5
    cosims = [float(np.dot(current_hs.ravel(), h.ravel()) / max(1e-10, np.linalg.norm(current_hs) * np.linalg.norm(h))) for h in history[-3:]]
    return float(np.mean(cosims)) if cosims else 0.5


def extract_raw_features(hidden_state: torch.Tensor, context_dim: int) -> np.ndarray:
    h = hidden_state.detach().float().cpu().numpy()
    features = np.concatenate([
        h.mean(axis=-1, keepdims=True), h.std(axis=-1, keepdims=True),
        np.percentile(h, 25, axis=-1, keepdims=True), np.percentile(h, 75, axis=-1, keepdims=True),
    ])
    f = features.ravel()
    if len(f) > context_dim:
        f = f[:context_dim]
    elif len(f) < context_dim:
        f = np.pad(f, (0, context_dim - len(f)))
    assert len(f) == context_dim
    return f.astype(np.float32)


@torch.no_grad()
def get_hidden_state(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    h = model.transformer.wte(input_ids)
    h = h + model.transformer.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
    for block in model.transformer.h:
        h = block(h)[0]
    return model.transformer.ln_f(h)


@torch.no_grad()
def run_ephemeral(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    cfg: FieldConfig,
    inject_field: bool = False,
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    lm_head = model.lm_head
    W_orig = lm_head.weight.detach().float().cpu().numpy().copy()
    out_f, in_f = W_orig.shape

    hl_cfg = HyperLoraConfig(
        context_dim=cfg.base_context_dim + (cfg.field_dim if inject_field else 0),
        rank=cfg.rank,
        in_features=in_f,
        out_features=out_f,
        scale_min=0.3,
        scale_max=1.2,
        seed=cfg.seed,
    )

    total_base, total_mod = 0.0, 0.0
    n_tokens = 0
    all_tok = []
    state_history: List[float] = []
    hs_history: List[np.ndarray] = []
    t0 = time.perf_counter()

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=cfg.max_length)
        input_ids = enc.input_ids.to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        logits_base = model(input_ids).logits
        nll_base_tok = torch.nn.functional.cross_entropy(
            logits_base[0, :-1].float(), input_ids[0, 1:], reduction="none"
        ).cpu().numpy()

        for pos in range(seq_len - 1):
            hs = get_hidden_state(model, input_ids[:, :pos + 1])
            h_pos = hs[0, pos].cpu().numpy().astype(np.float32)

            # Compute field signals
            entropy = compute_entropy(logits_base[0, pos])
            coherence = compute_coherence(h_pos, hs_history)
            hs_history.append(h_pos)
            if len(hs_history) > 20:
                hs_history.pop(0)

            prev_delta = state_history[-1] if state_history else 0.0

            # Build context with or without field injection
            raw_ctx = extract_raw_features(torch.from_numpy(h_pos), cfg.base_context_dim)
            if inject_field:
                field_vec = np.array([
                    entropy / 10.0,        # arousal (normalized)
                    coherence,              # coherence
                    max(-1.0, min(1.0, prev_delta * 5.0)),  # valence
                ], dtype=np.float32)
                ctx = np.concatenate([raw_ctx, field_vec])
            else:
                ctx = raw_ctx

            A, B, scale = hyperlora_generate(ctx, hl_cfg)

            # Apply
            W_np = lm_head.weight.detach().float().cpu().numpy()
            W_new, handle = apply_delta(W_np, A, B, scale)
            lm_head.weight.data = torch.from_numpy(W_new).to(device=device, dtype=lm_head.weight.dtype)

            # Forward
            logits_mod = model(input_ids[:, :pos + 1]).logits
            target = input_ids[0, pos + 1]
            nll_mod = torch.nn.functional.cross_entropy(logits_mod[0, -1].unsqueeze(0).float(), target.unsqueeze(0)).item()

            # Revert
            lm_head.weight.data = torch.from_numpy(revert_delta(W_new, handle)).to(device=device, dtype=lm_head.weight.dtype)

            nll_base = float(nll_base_tok[pos])
            nll_delta = nll_mod - nll_base
            total_base += nll_base
            total_mod += nll_mod
            state_history.append(nll_delta)
            n_tokens += 1

            all_tok.append({
                "pos": pos, "nll_base": nll_base, "nll_mod": nll_mod,
                "nll_delta": nll_delta, "scale": scale, "entropy": entropy,
                "coherence": coherence, "prev_delta": prev_delta,
            })

    dt = time.perf_counter() - t0
    drift = float(np.max(np.abs(lm_head.weight.detach().float().cpu().numpy() - W_orig)))
    deltas = [t["nll_delta"] for t in all_tok]
    return {
        "config": asdict(cfg),
        "inject_field": inject_field,
        "baseline_nll": total_base, "modified_nll": total_mod,
        "nll_delta": total_mod - total_base,
        "nll_delta_per_token": (total_mod - total_base) / max(1, n_tokens),
        "reversibility_drift": drift, "total_tokens": n_tokens,
        "total_time_s": dt, "time_per_token_ms": dt / max(1, n_tokens) * 1000,
        "per_token": all_tok,
        "fraction_improved": sum(1 for d in deltas if d < 0) / max(1, len(deltas)),
        "mean_scale": float(np.mean([t["scale"] for t in all_tok])),
        "tokens_degraded_gt_0_1": sum(1 for d in deltas if d > 0.1),
    }


def p(name, r):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  NLL delta:          {r['nll_delta']:+.6f}  ({r['nll_delta_per_token']:+.6f}/tok)")
    print(f"  Improved:           {r['fraction_improved']:.1%}")
    print(f"  Degraded >0.1:      {r['tokens_degraded_gt_0_1']}/{r['total_tokens']}")
    print(f"  Mean scale:         {r['mean_scale']:.4f}")
    print(f"  Reversibility:      {r['reversibility_drift']:.2e}")
    print(f"  Time:               {r['total_time_s']:.2f}s  ({r['time_per_token_ms']:.0f}ms/tok)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-json", default="quality/cognitive_field/v2_result.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    texts = ["The capital of France is", "Quantum computing relies on", "In the beginning, there was", "The theory of relativity shows", "Machine learning models can", "The history of Rome began"][:args.prompts]

    cfg = FieldConfig(model_name=args.model, rank=args.rank, max_length=args.max_length, prompts=args.prompts, seed=args.seed)

    log.info("Without field injection...")
    r1 = run_ephemeral(model, tokenizer, texts, cfg, inject_field=False)
    p("NO FIELD (raw context only)", r1)

    log.info("With field injection...")
    r2 = run_ephemeral(model, tokenizer, texts, cfg, inject_field=True)
    p("FIELD INJECTED (ctx + entropy + coherence + valence)", r2)

    d1, d2 = r1["nll_delta"], r2["nll_delta"]
    print(f"\n{'─'*60}")
    print(f"  FIELD EFFECT: ΔNLL {d1:+.4f} → {d2:+.4f}  (Δ{d2-d1:+.4f})")
    print(f"  IMPROVED:     {r1['fraction_improved']:.1%} → {r2['fraction_improved']:.1%}")
    print(f"{'─'*60}")

    combined = {"no_field": r1, "with_field": r2}
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info("Saved to %s", args.results_json)
