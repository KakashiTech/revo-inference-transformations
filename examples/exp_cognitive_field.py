"""REVO proto-campo cognitivo Φ/A/C — modulation by context.

Tests three cognitive signals that modulate the ephemeral delta:

  Φ (Valence):    Did the last delta improve NLL? If yes, amplify.
                  If no, dampen. A simple autoregressive feedback.

  A (Arousal):    Entropy of the logit distribution. High entropy →
                  model is uncertain → reduce scale (don't disturb a
                  fragile prediction). Low entropy → confident → can
                  accept more perturbation.

  C (Coherence):  Cosine similarity between current hidden state and
                  recent context window. High coherence → stable topic
                  → deltas are safer. Low coherence → topic shift →
                  reduce scale.

The cognitive field modulates the base HyperLoRA scale:
  scale_final = scale_base * Φ(prev_delta) * A(entropy) * C(coherence)
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._logging import get_logger
from revo.engine import apply_delta, revert_delta
from revo.hyperlora import HyperLoraConfig, hyperlora_generate

log = get_logger(__name__)


@dataclass
class CognitiveFieldConfig:
    model_name: str = "gpt2"
    rank: int = 8
    context_dim: int = 32
    max_length: int = 48
    prompts: int = 10
    seed: int = 0

    # Cognitive field gates (1.0 = neutral, >1 = amplify, <1 = dampen)
    valence_gain: float = 0.3
    """How much previous-token feedback modulates scale."""
    arousal_gain: float = 0.5
    """How much entropy modulates scale."""
    coherence_gain: float = 0.3
    """How much topic coherence modulates scale."""

    # Bounds
    scale_min: float = 0.1
    scale_max: float = 1.5
    entropy_threshold_low: float = 2.0
    entropy_threshold_high: float = 6.0
    coherence_window: int = 3
    """Number of past hidden states for coherence estimation."""


@dataclass
class FieldState:
    valence: float = 1.0
    """Running estimate: mean NLL delta over last N tokens. ∈ [-1, 1]"""
    arousal: float = 0.5
    """Current token entropy, normalized. ∈ [0, 1]"""
    coherence: float = 0.5
    """Topic stability, ∈ [0, 1]"""
    prev_nll_delta: float = 0.0
    """Last token's NLL delta (for valence)."""


def compute_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy of the logit distribution."""
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-10)), dim=-1)
    return float(entropy.mean().item())


def compute_coherence(
    current_hs: np.ndarray, history: List[np.ndarray]
) -> float:
    """Average cosine similarity between current and recent hidden states."""
    if not history:
        return 0.5
    cosims = []
    for h in history[-3:]:
        a = current_hs.ravel()
        b = h.ravel()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-10:
            cosims.append(0.0)
        else:
            cosims.append(float(np.dot(a, b) / denom))
    return float(np.mean(cosims)) if cosims else 0.5


def modulate_scale(
    cfg: CognitiveFieldConfig,
    state: FieldState,
    base_scale: float,
    entropy: float,
) -> float:
    """Apply cognitive field modulation to the base scale.

    Modulates scale: lower when uncertain (high entropy, low coherence)
    or when previous delta degraded quality (negative valence).
    """
    # Arousal: entropy gate
    if entropy < cfg.entropy_threshold_low:
        a_gate = 1.0 + cfg.arousal_gain * 0.5
    elif entropy > cfg.entropy_threshold_high:
        a_gate = 1.0 - cfg.arousal_gain * 0.5
    else:
        ratio = (entropy - cfg.entropy_threshold_low) / (
            cfg.entropy_threshold_high - cfg.entropy_threshold_low
        )
        a_gate = 1.0 - cfg.arousal_gain * ratio * 0.5

    # Coherence: stable topic → amplify, topic shift → dampen
    c_gate = 1.0 + cfg.coherence_gain * (state.coherence - 0.5) * 2.0

    # Valence: previous delta improved NLL → amplify
    v_gate = 1.0 + cfg.valence_gain * max(-1.0, min(1.0, -state.prev_nll_delta * 10.0))

    scale = base_scale * a_gate * c_gate * v_gate
    scale = max(cfg.scale_min, min(cfg.scale_max, scale))
    return scale


@torch.no_grad()
def get_hidden_state(
    model: nn.Module, input_ids: torch.Tensor
) -> torch.Tensor:
    """Get hidden state from last layer (before lm_head)."""
    h = model.transformer.wte(input_ids)
    h = h + model.transformer.wpe(
        torch.arange(input_ids.shape[1], device=input_ids.device)
    )
    for block in model.transformer.h:
        h = block(h)[0]
    return model.transformer.ln_f(h)


def extract_context_features(
    hidden_state: torch.Tensor, token_idx: int, context_dim: int = 32
) -> np.ndarray:
    """Exactly context_dim floats from residual stream."""
    h = hidden_state.detach().float().cpu().numpy()
    features = np.concatenate([
        h.mean(axis=-1, keepdims=True),
        h.std(axis=-1, keepdims=True),
        np.percentile(h, 25, axis=-1, keepdims=True),
        np.percentile(h, 75, axis=-1, keepdims=True),
        np.max(h, axis=-1, keepdims=True),
        np.min(h, axis=-1, keepdims=True),
        np.array([token_idx / 1024], dtype=np.float32),
    ])
    f = features.ravel()
    if len(f) > context_dim:
        f = f[:context_dim]
    elif len(f) < context_dim:
        f = np.pad(f, (0, context_dim - len(f)))
    assert len(f) == context_dim
    return f.astype(np.float32)


@torch.no_grad()
def run_with_cognitive_field(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    cfg: CognitiveFieldConfig,
    use_field: bool = False,
) -> Dict[str, Any]:
    """Run ephemeral cycle with optional cognitive field modulation."""
    model.eval()
    device = next(model.parameters()).device
    lm_head = model.lm_head

    # Baseline forward once
    total_baseline = 0.0
    total_modified = 0.0
    n_tokens = 0
    all_tokens: List[Dict[str, Any]] = []
    state = FieldState()
    hs_history: List[np.ndarray] = []
    t_start = time.perf_counter()

    W_orig = lm_head.weight.detach().float().cpu().numpy().copy()
    out_f, in_f = W_orig.shape

    hl_cfg = HyperLoraConfig(
        context_dim=cfg.context_dim,
        rank=cfg.rank,
        in_features=in_f,
        out_features=out_f,
        scale_min=0.1,
        scale_max=1.5,
        seed=cfg.seed,
    )

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=cfg.max_length)
        input_ids = enc.input_ids.to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        # Baseline logits (one forward pass)
        logits_base = model(input_ids).logits
        nll_per_token = torch.nn.functional.cross_entropy(
            logits_base[0, :-1].float(),
            input_ids[0, 1:],
            reduction="none",
        ).cpu().numpy()

        for pos in range(seq_len - 1):
            # Get hidden state at this position
            hs = get_hidden_state(model, input_ids[:, :pos + 1])
            h_pos = hs[0, pos].cpu().numpy().astype(np.float32)

            # Compute entropy from baseline logits
            baseline_logits_at_pos = logits_base[0, pos]
            entropy = compute_entropy(baseline_logits_at_pos)

            # Compute coherence
            coherence = compute_coherence(h_pos, hs_history)
            hs_history.append(h_pos)
            if len(hs_history) > 20:
                hs_history.pop(0)

            # Build context
            ctx = extract_context_features(
                torch.from_numpy(h_pos), pos, context_dim=cfg.context_dim
            )

            # Generate delta (deterministic from context)
            A, B, base_scale = hyperlora_generate(ctx, hl_cfg)

            # Modulate scale by cognitive field
            if use_field:
                scale = modulate_scale(cfg, state, base_scale, entropy)
            else:
                scale = base_scale

            # Apply delta
            W_np = lm_head.weight.detach().float().cpu().numpy()
            W_new, handle = apply_delta(W_np, A, B, scale)
            lm_head.weight.data = torch.from_numpy(W_new).to(
                device=device, dtype=lm_head.weight.dtype
            )

            # Forward with modified head
            logits_mod = model(input_ids[:, :pos + 1]).logits
            target = input_ids[0, pos + 1]
            nll_mod = torch.nn.functional.cross_entropy(
                logits_mod[0, -1].unsqueeze(0).float(),
                target.unsqueeze(0),
            ).item()

            # Revert
            W_revert = revert_delta(W_new, handle)
            lm_head.weight.data = torch.from_numpy(W_revert).to(
                device=device, dtype=lm_head.weight.dtype
            )

            nll_base = float(nll_per_token[pos])
            nll_delta = nll_mod - nll_base
            total_baseline += nll_base
            total_modified += nll_mod
            n_tokens += 1

            state.prev_nll_delta = nll_delta
            state.coherence = coherence
            state.arousal = entropy / 10.0

            all_tokens.append({
                "pos": pos,
                "nll_base": nll_base,
                "nll_mod": nll_mod,
                "nll_delta": nll_delta,
                "scale": scale,
                "base_scale": base_scale,
                "entropy": entropy,
                "coherence": coherence,
                "valence_gate": (1.0 + cfg.valence_gain * max(-1.0, min(1.0, -nll_delta * 10.0))),
            })

    t_total = time.perf_counter() - t_start

    # Reversibility check
    drift = float(np.max(np.abs(lm_head.weight.detach().float().cpu().numpy() - W_orig)))

    deltas = [t["nll_delta"] for t in all_tokens]
    return {
        "config": asdict(cfg),
        "use_field": use_field,
        "baseline_nll": total_baseline,
        "modified_nll": total_modified,
        "nll_delta": total_modified - total_baseline,
        "nll_delta_per_token": (total_modified - total_baseline) / max(1, n_tokens),
        "reversibility_drift": drift,
        "total_tokens": n_tokens,
        "total_time_s": t_total,
        "time_per_token_ms": t_total / max(1, n_tokens) * 1000,
        "per_token": all_tokens,
        "tokens_improved": sum(1 for d in deltas if d < 0),
        "tokens_degraded_gt_0_1": sum(1 for d in deltas if d > 0.1),
        "fraction_improved": sum(1 for d in deltas if d < 0) / max(1, len(deltas)),
        "mean_scale": float(np.mean([t["scale"] for t in all_tokens])),
        "mean_entropy": float(np.mean([t["entropy"] for t in all_tokens])),
        "mean_coherence": float(np.mean([t["coherence"] for t in all_tokens])),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="REVO Cognitive Field Experiment")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--context-dim", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-json", default="quality/cognitive_field/result.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    texts = [
        "The capital of France is",
        "Quantum computing relies on",
        "In the beginning, there was",
        "The theory of relativity shows",
        "Machine learning models can",
        "The history of Rome began",
        "Neural networks are composed of",
        "The speed of light is",
        "In mathematics, a prime number",
        "The human brain processes",
    ][:args.prompts]

    cfg = CognitiveFieldConfig(
        model_name=args.model,
        rank=args.rank,
        context_dim=args.context_dim,
        max_length=args.max_length,
        prompts=args.prompts,
        seed=args.seed,
    )

    log.info("Running WITHOUT cognitive field (baseline deltas)...")
    result_no_field = run_with_cognitive_field(
        model, tokenizer, texts, cfg, use_field=False
    )

    log.info("Running WITH cognitive field modulation...")
    result_with_field = run_with_cognitive_field(
        model, tokenizer, texts, cfg, use_field=True
    )

    # Summary
    def _summary(name, r):
        lines = [
            f"\n{'='*60}",
            f"  {name}",
            f"{'='*60}",
            f"  NLL delta:          {r['nll_delta']:+.6f}  ({r['nll_delta_per_token']:+.6f}/tok)",
            f"  Tokens improved:    {r['tokens_improved']}/{r['total_tokens']} ({r['fraction_improved']:.1%})",
            f"  Degraded >0.1:      {r['tokens_degraded_gt_0_1']}/{r['total_tokens']}",
            f"  Mean scale:         {r['mean_scale']:.4f}",
            f"  Mean entropy:       {r['mean_entropy']:.4f}",
            f"  Mean coherence:     {r['mean_coherence']:.4f}",
            f"  Reversibility:      {r['reversibility_drift']:.2e}",
            f"  Time:               {r['total_time_s']:.2f}s ({r['time_per_token_ms']:.0f}ms/tok)",
        ]
        return "\n".join(lines)

    print(_summary("BASELINE (no field)", result_no_field))
    print(_summary("COGNITIVE FIELD", result_with_field))

    delta_no = result_no_field["nll_delta"]
    delta_with = result_with_field["nll_delta"]
    impr_no = result_no_field["fraction_improved"]
    impr_with = result_with_field["fraction_improved"]

    print(f"\n{'─'*60}")
    print(f"  FIELD EFFECT: ΔNLL {delta_no:+.4f} → {delta_with:+.4f} ({delta_with - delta_no:+.4f})")
    print(f"  IMPROVED:     {impr_no:.1%} → {impr_with:.1%}")
    print(f"{'─'*60}\n")

    # Save both
    combined = {
        "no_field": result_no_field,
        "with_field": result_with_field,
        "comparison": {
            "nll_delta_improvement": delta_with - delta_no,
            "fraction_improved_no_field": impr_no,
            "fraction_improved_with_field": impr_with,
        },
    }
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info("Saved to %s", args.results_json)
