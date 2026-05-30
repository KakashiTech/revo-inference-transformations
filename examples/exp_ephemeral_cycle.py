"""REVO ephemeral cycle per-token: reconstruct → apply → generate → revert.

This is the core loop of the vision. For each token in a forward pass:
  1. Extract context features from residual stream
  2. Generate HyperLoRA delta for lm_head (deterministic from context)
  3. Apply the delta (modify lm_head.weight ephemerally)
  4. Run forward through lm_head to get logits
  5. Revert the delta (restore original weight)
  6. Measure NLL impact

The model never has modified weights for more than one forward call.
This is the "model as trajectory" — not a single static object.
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._logging import get_logger
from revo.engine import apply_delta, revert_delta
from revo.hyperlora import HyperLoraConfig, hyperlora_generate
from revo.mode_cache import ModeCache

log = get_logger(__name__)


@dataclass
class EphemeralCycleConfig:
    model_name: str = "gpt2"
    rank: int = 8
    context_dim: int = 32
    max_length: int = 64
    prompts: int = 10
    seed: int = 0
    scale_min: float = 0.3
    scale_max: float = 1.2
    use_mode_cache: bool = True
    mode_cache_ttl: int = 900
    mode_cache_max: int = 64
    """Number of apply/revert cycles for reversibility audit."""
    audit_cycles: int = 5


@dataclass
class TokenMetrics:
    token_idx: int
    token_id: int
    token_str: str
    nll_baseline: float
    nll_modified: float
    nll_delta: float
    scale: float
    context_norm: float
    apply_time_ms: float


@dataclass
class CycleResult:
    config: Dict[str, Any]
    baseline_nll: float
    modified_nll: float
    nll_delta: float
    time_ratio: float
    per_token_metrics: List[Dict[str, Any]]
    reversibility_drift: float
    mode_cache_hits: int
    mode_cache_misses: int
    total_tokens: int
    total_time_s: float


def extract_context_features(
    hidden_state: torch.Tensor, token_idx: int, context_dim: int = 32
) -> np.ndarray:
    """Build a context vector of exactly `context_dim` from the residual stream."""
    h = hidden_state.detach().float().cpu().numpy()
    # Use quantiles for richer signal
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
    assert len(f) == context_dim, f"expected {context_dim}, got {len(f)}"
    return f.astype(np.float32)


@torch.no_grad()
def get_hidden_state(
    model: nn.Module, input_ids: torch.Tensor, layer_idx: int = -1
) -> torch.Tensor:
    """Get hidden state from a specific layer (default: last before lm_head)."""
    # GPT-2: transformer.h[layer_idx] output, then ln_f
    if layer_idx < 0:
        layer_idx = len(model.transformer.h) + layer_idx
    h = model.transformer.wte(input_ids)
    h = h + model.transformer.wpe(
        torch.arange(input_ids.shape[1], device=input_ids.device)
    )
    for i, block in enumerate(model.transformer.h):
        h = block(h)[0]
        if i == layer_idx:
            break
    h = model.transformer.ln_f(h)
    return h


@torch.no_grad()
def run_ephemeral_cycle(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    config: EphemeralCycleConfig,
) -> CycleResult:
    """Run the full REVO ephemeral cycle on a set of texts.

    For each token in each text:
      1. Get hidden state from last layer
      2. Extract context features
      3. Generate HyperLoRA delta (or get from cache)
      4. Apply delta to lm_head
      5. Compute logits with modified head
      6. Compute NLL for the target token
      7. Revert delta
      8. Record metrics

    Returns aggregated CycleResult with per-token breakdown.
    """
    model.eval()
    device = next(model.parameters()).device
    cfg = config

    # Determine lm_head orientation
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.transformer.wte
    W_orig = lm_head.weight.detach().float().cpu().numpy().copy()
    out_f, in_f = W_orig.shape

    # Build HyperLoRA config for lm_head
    hl_cfg = HyperLoraConfig(
        context_dim=cfg.context_dim,
        rank=cfg.rank,
        in_features=in_f,
        out_features=out_f,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
        seed=cfg.seed,
    )

    # Mode cache
    mode_cache: Optional[ModeCache] = None
    if cfg.use_mode_cache:
        mode_cache = ModeCache(
            ttl_seconds=cfg.mode_cache_ttl,
            max_size=cfg.mode_cache_max,
        )

    # Results
    all_metrics: List[TokenMetrics] = []
    total_nll_baseline = 0.0
    total_nll_modified = 0.0
    cache_hits = 0
    cache_misses = 0
    t_start = time.perf_counter()
    n_tokens_total = 0

    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=cfg.max_length
        )
        input_ids = enc.input_ids.to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        # Baseline: forward with original lm_head
        logits_base = model(input_ids).logits  # (1, seq_len, vocab)
        baseline_nll = torch.nn.functional.cross_entropy(
            logits_base[0, :-1].float(),
            input_ids[0, 1:],
            reduction="none",
        ).cpu().numpy()  # per token

        # For each token, apply ephemeral delta and measure
        for pos in range(seq_len - 1):
            # 1. Get hidden state at this position (before lm_head)
            hs = get_hidden_state(
                model, input_ids[:, : pos + 1], layer_idx=-1
            )  # (1, pos+1, hidden)
            h_pos = hs[0, pos]  # (hidden,)

            # 2. Extract context features
            ctx = extract_context_features(h_pos, pos, context_dim=cfg.context_dim)

            # 3. Generate or retrieve delta
            mode_key = f"pos_{pos}_ctx_{hash(ctx.tobytes()) & 0xFFFFFFFF}"
            if mode_cache is not None:
                cached = mode_cache.get(mode_key)
                if cached is not None:
                    A, B, scale = cached
                    cache_hits += 1
                else:
                    A, B, scale = hyperlora_generate(ctx, hl_cfg)
                    mode_cache.put(mode_key, (A, B, scale))
                    cache_misses += 1
            else:
                A, B, scale = hyperlora_generate(ctx, hl_cfg)

            # 4. Apply delta to lm_head
            W_np = lm_head.weight.detach().float().cpu().numpy()
            t_apply = time.perf_counter()
            W_new, handle = apply_delta(W_np, A, B, scale)
            lm_head.weight.data = torch.from_numpy(W_new).to(
                device=device, dtype=lm_head.weight.dtype
            )
            t_apply_ms = (time.perf_counter() - t_apply) * 1000

            # 5. Forward with modified head (only last position)
            logits_at_pos = model(input_ids[:, : pos + 1]).logits
            # NLL for this position: pr(target token at pos+1 | context)
            target = input_ids[0, pos + 1]
            nll_mod = torch.nn.functional.cross_entropy(
                logits_at_pos[0, -1].unsqueeze(0).float(),
                target.unsqueeze(0),
            ).item()

            # 6. Revert delta
            W_revert = revert_delta(W_new, handle)
            lm_head.weight.data = torch.from_numpy(W_revert).to(
                device=device, dtype=lm_head.weight.dtype
            )

            # 7. Record
            nll_base_tok = float(baseline_nll[pos])
            token_id = int(input_ids[0, pos + 1].item())
            token_str = tokenizer.decode([token_id])
            all_metrics.append(TokenMetrics(
                token_idx=n_tokens_total,
                token_id=token_id,
                token_str=token_str,
                nll_baseline=nll_base_tok,
                nll_modified=nll_mod,
                nll_delta=nll_mod - nll_base_tok,
                scale=scale,
                context_norm=float(np.linalg.norm(ctx)),
                apply_time_ms=t_apply_ms,
            ))
            total_nll_baseline += nll_base_tok
            total_nll_modified += nll_mod
            n_tokens_total += 1

    t_total = time.perf_counter() - t_start

    # — Reversibility audit —
    # Run N cycles of apply/revert on random deltas, measure final drift
    drift = audit_reversibility(model, cfg.audit_cycles, hl_cfg)
    W_final = lm_head.weight.detach().float().cpu().numpy()
    final_drift = float(np.max(np.abs(W_final - W_orig)))

    nll_delta = total_nll_modified - total_nll_baseline
    return CycleResult(
        config=asdict(cfg),
        baseline_nll=total_nll_baseline,
        modified_nll=total_nll_modified,
        nll_delta=nll_delta,
        time_ratio=t_total / max(1e-9, t_total),
        per_token_metrics=[asdict(m) for m in all_metrics],
        reversibility_drift=final_drift,
        mode_cache_hits=cache_hits,
        mode_cache_misses=cache_misses,
        total_tokens=n_tokens_total,
        total_time_s=t_total,
    )


@torch.no_grad()
def audit_reversibility(
    model: nn.Module, cycles: int, hl_cfg: HyperLoraConfig
) -> float:
    """Apply and revert N random deltas, report max drift from original.

    The audit measures:
      - Max absolute weight drift after N cycles
      - Whether drift accumulates or stays bounded
    """
    lm_head = model.lm_head if hasattr(model, "lm_head") else model.transformer.wte
    W_ref = lm_head.weight.detach().float().cpu().numpy().copy()
    max_drift = 0.0

    for cyc in range(cycles):
        rng = np.random.default_rng(cyc)
        ctx = rng.standard_normal(hl_cfg.context_dim).astype(np.float32)
        A, B, scale = hyperlora_generate(ctx, hl_cfg)
        W_np = lm_head.weight.detach().float().cpu().numpy()
        W_modified, handle = apply_delta(W_np, A, B, scale)
        lm_head.weight.data = torch.from_numpy(W_modified).to(
            device=lm_head.weight.device, dtype=lm_head.weight.dtype
        )
        W_reverted = revert_delta(W_modified, handle)
        lm_head.weight.data = torch.from_numpy(W_reverted).to(
            device=lm_head.weight.device, dtype=lm_head.weight.dtype
        )
        drift = float(np.max(np.abs(
            lm_head.weight.detach().float().cpu().numpy() - W_ref
        )))
        max_drift = max(max_drift, drift)
        log.info("Audit cycle %d/%d: max drift = %.2e", cyc + 1, cycles, drift)

    # Restore exact original
    lm_head.weight.data = torch.from_numpy(W_ref).to(
        device=lm_head.weight.device, dtype=lm_head.weight.dtype
    )
    return max_drift


def summarize(result: CycleResult) -> str:
    """Human-readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("REVO EPHEMERAL CYCLE — RESULTS")
    lines.append("=" * 60)
    lines.append(f"Model:                    {result.config['model_name']}")
    lines.append(f"Rank:                     {result.config['rank']}")
    lines.append(f"Context dim:              {result.config['context_dim']}")
    lines.append(f"Total tokens:             {result.total_tokens}")
    lines.append(f"ModeCache:                {'ON' if result.config['use_mode_cache'] else 'OFF'}")
    if result.config['use_mode_cache']:
        lines.append(f"  Hits:                   {result.mode_cache_hits}")
        lines.append(f"  Misses:                 {result.mode_cache_misses}")
        hr = result.mode_cache_hits / max(1, result.mode_cache_hits + result.mode_cache_misses)
        lines.append(f"  Hit rate:               {hr:.2%}")
    lines.append("")
    lines.append("NLL METRICS")
    lines.append(f"  Baseline:               {result.baseline_nll:.4f}")
    lines.append(f"  Modified:               {result.modified_nll:.4f}")
    lines.append(f"  NLL delta:              {result.nll_delta:+.6f}")
    lines.append(f"  Delta per token:        {result.nll_delta / max(1, result.total_tokens):+.6f}")
    lines.append("")
    lines.append(f"Reversibility drift:      {result.reversibility_drift:.2e}")
    lines.append(f"Total time:               {result.total_time_s:.3f}s")
    lines.append("")

    # Per-token statistics
    if result.per_token_metrics:
        deltas = [m["nll_delta"] for m in result.per_token_metrics]
        scales = [m["scale"] for m in result.per_token_metrics]
        lines.append("PER-TOKEN STATS")
        lines.append(f"  Max NLL delta:          {max(deltas):+.6f}")
        lines.append(f"  Min NLL delta:          {min(deltas):+.6f}")
        lines.append(f"  Mean NLL delta:         {sum(deltas) / len(deltas):+.6f}")
        lines.append(f"  Median NLL delta:       {sorted(deltas)[len(deltas)//2]:+.6f}")
        lines.append(f"  Fraction improved:      {sum(1 for d in deltas if d < 0) / len(deltas):.2%}")
        lines.append(f"  Fraction degraded >0.1: {sum(1 for d in deltas if d > 0.1) / len(deltas):.2%}")
        lines.append(f"  Mean scale:             {sum(scales) / len(scales):.4f}")
        lines.append(f"  Scale range:            [{min(scales):.4f}, {max(scales):.4f}]")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_result(result: CycleResult, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    log.info("Saved result to %s", path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="REVO Ephemeral Cycle Experiment")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--context-dim", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mode-cache", action="store_true")
    ap.add_argument("--audit-cycles", type=int, default=10)
    ap.add_argument("--results-json", default="quality/ephemeral/cycle_result.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info("Loading model %s ...", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    cfg = EphemeralCycleConfig(
        model_name=args.model,
        rank=args.rank,
        context_dim=args.context_dim,
        max_length=args.max_length,
        prompts=args.prompts,
        seed=args.seed,
        use_mode_cache=not args.no_mode_cache,
        audit_cycles=args.audit_cycles,
    )

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

    log.info("Running ephemeral cycle with %d prompts...", len(texts))
    result = run_ephemeral_cycle(model, tokenizer, texts, cfg)

    print(summarize(result))
    save_result(result, args.results_json)
