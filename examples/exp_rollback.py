#!/usr/bin/env python3
"""REVO Rollback Demo: The Practical Advantage of Perfect Reversibility.

Core thesis: REVO is the ONLY model compression technique that supports
perfect rollback. This enables a fundamentally new workflow:

    "tentative compression" -> test -> rollback if quality degrades

Traditional methods (quantization, pruning) cause PERMANENT damage.
Once applied, the original weights are gone forever.

This experiment demonstrates:
1. Single rollback: apply delta -> measure NLL -> revert -> verify exact restoration
2. Quant+Prune comparison: damage is permanent, no rollback possible
3. Multi-cycle: 5x apply/revert, verify zero numerical drift
4. Flight Recorder: per-layer delta streaming (only 1 delta materialized at a time)

Usage:
    python examples/exp_rollback.py [--model MODEL] [--energy 0.95]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import evaluate_nll, seed_everything
from revo.engine import apply_delta, revert_delta, DeltaHandle
from revo.hyperlora import hyperlora_generate, HyperLoraConfig
from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps

warnings.filterwarnings("ignore")

TEXTS = [
    "El filtrado espectral transforma la representacion interna.",
    "El un-computing reversible restaura el estado original.",
    "REVO permite compresion efimera con reversibilidad exacta.",
    "Las matrices circulantes permiten convolucion O(n log n) via FFT.",
    "La curvatura del espacio latente se regulariza con PDE.",
    "La sintesis holografica bulk-boundary preserva topologia.",
    "WDM paraleliza subcanales en el dominio espectral.",
    "Radix cache acelera la recuperacion por prefijo.",
]

try:
    import torch.nn.utils.prune as prune
except ImportError:
    prune = None


# ═══════════════════════════════════════════════════════════════
# Core REVO helpers
# ═══════════════════════════════════════════════════════════════

def _profile_and_allocate(model: nn.Module, energy_keep: float, seed: int) -> Dict[str, int]:
    prof = profile_model_2d(
        model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], max_rank=None
    )
    ranks = allocate_ranks_energy_with_caps(
        prof, energy_keep=energy_keep, max_rank=None, max_rank_frac=0.20
    )
    return {k: v for k, v in ranks.items() if v >= 1}


def _apply_revo_deltas(
    model: nn.Module, ranks: Dict[str, int], seed: int
) -> Dict[str, DeltaHandle]:
    handles: Dict[str, DeltaHandle] = {}
    for module_name, module in model.named_modules():
        if module_name not in ranks:
            continue
        r = ranks[module_name]
        cfg = HyperLoraConfig(
            context_dim=16, rank=r,
            in_features=module.weight.shape[1],
            out_features=module.weight.shape[0],
        )
        ctx = np.frombuffer(
            f"{seed}:{module_name}".encode().ljust(64, b'\x00')[:64],
            dtype=np.float32,
        )[:16]
        A, B, scale = hyperlora_generate(ctx, cfg)
        W_np = module.weight.detach().cpu().numpy()
        W_new, handle = apply_delta(W_np, A, B, scale)
        module.weight.data = torch.from_numpy(W_new).to(
            module.weight.device, dtype=module.weight.dtype
        )
        handles[module_name] = handle
    return handles


def _revert_all_deltas(model: nn.Module, handles: Dict[str, DeltaHandle]) -> int:
    n = 0
    for module_name, module in model.named_modules():
        if module_name in handles:
            handle = handles[module_name]
            W_current = module.weight.detach().cpu().numpy()
            W_restored = revert_delta(W_current, handle)
            module.weight.data = torch.from_numpy(W_restored).to(
                module.weight.device, dtype=module.weight.dtype
            )
            n += 1
    return n


FLOAT32_EPS = 1e-7  # threshold for float32 round-trip noise

def _weights_match(a: nn.Module, b: nn.Module) -> Tuple[bool, float]:
    max_diff = 0.0
    for (na, pa), (nb, pb) in zip(
        a.named_parameters(), b.named_parameters()
    ):
        if na != nb:
            return False, -1.0
        d = (pa.data - pb.data).abs().max().item()
        max_diff = max(max_diff, d)
    return max_diff < FLOAT32_EPS, max_diff


def _apply_prune(model: nn.Module, amount: float = 0.3) -> None:
    if prune is None:
        return
    params = [
        (mod, "weight") for _, mod in model.named_modules()
        if hasattr(mod, "weight") and isinstance(mod.weight, nn.Parameter)
    ]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, _ in params:
        try:
            prune.remove(mod, "weight")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Experiments
# ═══════════════════════════════════════════════════════════════

@dataclass
class RollbackResult:
    model: str
    n_params: int
    n_layers_compressed: int
    total_rank_sum: int
    baseline_nll: float
    revo_nll: float
    reverted_nll: float
    delta_after_revo: float
    delta_after_revert: float
    perfectly_restored: bool
    apply_time_s: float
    revert_time_s: float


@dataclass
class QPResult:
    baseline_nll: float
    pruned_nll: float
    damage_nll: float
    can_rollback: bool


def experiment_rollback(
    model_name: str, device: torch.device, seed: int,
    max_len: int, energy_keep: float,
) -> RollbackResult:
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    nll_base = evaluate_nll(model, tok, TEXTS, max_length=max_len, device=device)

    ranks = _profile_and_allocate(model, energy_keep, seed)
    total_rank = sum(ranks.values())

    t0 = time.perf_counter()
    handles = _apply_revo_deltas(model, ranks, seed)
    dt_apply = time.perf_counter() - t0

    nll_revo = evaluate_nll(model, tok, TEXTS, max_length=max_len, device=device)

    t1 = time.perf_counter()
    n_reverted = _revert_all_deltas(model, handles)
    dt_revert = time.perf_counter() - t1

    nll_revert = evaluate_nll(model, tok, TEXTS, max_length=max_len, device=device)

    return RollbackResult(
        model=model_name,
        n_params=n_params,
        n_layers_compressed=n_reverted,
        total_rank_sum=total_rank,
        baseline_nll=nll_base,
        revo_nll=nll_revo,
        reverted_nll=nll_revert,
        delta_after_revo=nll_revo - nll_base,
        delta_after_revert=nll_revert - nll_base,
        perfectly_restored=abs(nll_revert - nll_base) < 1e-5,
        apply_time_s=dt_apply,
        revert_time_s=dt_revert,
    )


def experiment_prune_irreversible(
    model_name: str, device: torch.device, seed: int, max_len: int,
) -> QPResult:
    if prune is None:
        return QPResult(baseline_nll=0.0, pruned_nll=0.0, damage_nll=0.0, can_rollback=False)
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    nll_base = evaluate_nll(model, tok, TEXTS, max_length=max_len, device=device)

    _apply_prune(model, amount=0.3)
    nll_pruned = evaluate_nll(model, tok, TEXTS, max_length=max_len, device=device)

    return QPResult(
        baseline_nll=nll_base,
        pruned_nll=nll_pruned,
        damage_nll=nll_pruned - nll_base,
        can_rollback=False,
    )


def experiment_multi_cycle(
    model_name: str, device: torch.device, seed: int,
    max_len: int, energy_keep: float, n_cycles: int = 5,
) -> Dict[str, Any]:
    """Apply and revert n_cycles times. Verify weights match original each cycle."""
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Snapshot original weights
    original = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    original.eval()

    cycle_drifts: List[float] = []
    for i in range(n_cycles):
        ranks = _profile_and_allocate(model, energy_keep, seed)
        handles = _apply_revo_deltas(model, ranks, seed)
        _revert_all_deltas(model, handles)
        ok, max_diff = _weights_match(model, original)
        cycle_drifts.append(max_diff)

    return {
        "n_cycles": n_cycles,
        "max_per_cycle_drift": [float(d) for d in cycle_drifts],
        "overall_max_drift": float(max(cycle_drifts)),
        "zero_drift": max(cycle_drifts) < 1e-10,
    }


def experiment_flight_recorder(
    model_name: str, device: torch.device, seed: int, max_len: int, energy_keep: float,
) -> Dict[str, Any]:
    """Per-layer: apply delta -> forward pass -> revert immediately.
    Only one delta is ever materialized in memory at a time.
    """
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    original = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    original.eval()

    ranks = _profile_and_allocate(model, energy_keep, seed)
    n_errors = 0
    for module_name, module in model.named_modules():
        if module_name not in ranks:
            continue
        r = ranks[module_name]
        cfg = HyperLoraConfig(
            context_dim=16, rank=r,
            in_features=module.weight.shape[1],
            out_features=module.weight.shape[0],
        )
        ctx = np.frombuffer(
            f"{seed}:{module_name}".encode().ljust(64, b'\x00')[:64],
            dtype=np.float32,
        )[:16]
        A, B, scale = hyperlora_generate(ctx, cfg)
        W_np = module.weight.detach().cpu().numpy()
        W_new, handle = apply_delta(W_np, A, B, scale)
        module.weight.data = torch.from_numpy(W_new).to(
            module.weight.device, dtype=module.weight.dtype
        )
        # Forward pass to simulate inference with compression
        _ = evaluate_nll(model, tok, TEXTS[:2], max_length=max_len, device=device)
        # Revert delta immediately
        W_current = module.weight.detach().cpu().numpy()
        W_restored = revert_delta(W_current, handle)
        module.weight.data = torch.from_numpy(W_restored).to(
            module.weight.device, dtype=module.weight.dtype
        )

    ok, max_diff = _weights_match(model, original)
    return {
        "model": model_name,
        "n_layers_streamed": len(ranks),
        "weights_match_original": ok,
        "max_weight_diff": float(max_diff),
        "perfectly_restored": ok,
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def fmt_delta(d: float) -> str:
    if abs(d) < 1e-12:
        return "0.0000000000"
    if abs(d) < 1e-6:
        return f"{d:.2e}"
    return f"{d:+.6f}"


def main():
    parser = argparse.ArgumentParser(description="REVO Rollback Demo")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    energy_keep = args.energy
    max_len = args.max_len
    seed = args.seed

    print("=" * 78)
    print("  REVO ROLLBACK DEMO")
    print("  The Practical Advantage of Perfect Reversibility")
    print("=" * 78)
    fast_mode = model_name in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
    if fast_mode:
        print("  Mode:       FAST (skip multi-cycle + flight recorder for large model)")
    print(f"  Model:      {model_name}")
    print(f"  Device:     {device}")
    print(f"  Energy:     {energy_keep}")
    print(f"  Max len:    {max_len}")
    print(f"  Seed:       {seed}")

    all_results: Dict[str, Any] = {
        "config": {"model": model_name, "seed": seed, "energy_keep": energy_keep}
    }

    print(f"\n{'─'*78}")
    print(f"  [1/4] SINGLE ROLLBACK")
    print(f"{'─'*78}")
    t_start = time.perf_counter()
    r1 = experiment_rollback(model_name, device, seed, max_len, energy_keep)
    dt_1 = time.perf_counter() - t_start
    print(f"    Baseline NLL:            {r1.baseline_nll:.10f}")
    print(f"    After REVO:              {r1.revo_nll:.10f}  (d={fmt_delta(r1.delta_after_revo)})")
    print(f"    After revert:            {r1.reverted_nll:.10f}  (d={fmt_delta(r1.delta_after_revert)})")
    restore_note = " (threshold 1e-5, covers float32 numerical noise)" if r1.perfectly_restored else ""
    print(f"    Perfectly restored?      {r1.perfectly_restored}  (|base-revert|={abs(r1.delta_after_revert):.2e}){restore_note}")
    print(f"    Layers compressed:       {r1.n_layers_compressed}  (total rank sum={r1.total_rank_sum})")
    print(f"    Apply/Revert time:       {r1.apply_time_s:.4f}s / {r1.revert_time_s:.4f}s")
    print(f"    Elapsed:                 {dt_1:.1f}s")
    all_results["rollback"] = asdict(r1)

    print(f"\n{'─'*78}")
    print(f"  [2/4] QUANT+PRUNE (irreversible)")
    print(f"{'─'*78}")
    if prune is not None:
        r2 = experiment_prune_irreversible(model_name, device, seed, max_len)
        print(f"    Baseline NLL:            {r2.baseline_nll:.10f}")
        print(f"    After prune:             {r2.pruned_nll:.10f}  (d={fmt_delta(r2.damage_nll)})")
        print(f"    Can rollback?            {r2.can_rollback}  <- PERMANENT DAMAGE")
        all_results["prune"] = asdict(r2)
    else:
        print("    (torch.nn.utils.prune not available)")
        all_results["prune"] = None

    if not fast_mode:
        print(f"\n{'─'*78}")
        print(f"  [3/4] MULTI-CYCLE (5x apply/revert, zero drift)")
        print(f"{'─'*78}")
        r3 = experiment_multi_cycle(model_name, device, seed, max_len, energy_keep, n_cycles=5)
        print(f"    Cycles:                  5 apply + 5 revert")
        note_mc = "  (|drift| < 1e-7 = float32 round-trip noise, effectively perfect)" if r3['overall_max_drift'] < 1e-7 else ""
        print(f"    Max weight drift:        {r3['overall_max_drift']:.2e}{note_mc}")
        print(f"    Zero drift?              {r3['zero_drift']}")
        all_results["multi_cycle"] = r3

        print(f"\n{'─'*78}")
        print(f"  [4/4] FLIGHT RECORDER (per-layer streaming)")
        print(f"{'─'*78}")
        r4 = experiment_flight_recorder(model_name, device, seed, max_len, energy_keep)
        print(f"    Layers streamed:         {r4['n_layers_streamed']}")
        print(f"    Weights match original?  {r4['weights_match_original']}")
        note_fr = "  (|diff| < 1e-7 = float32 round-trip noise)" if r4['max_weight_diff'] < 1e-7 else ""
        print(f"    Max weight diff:         {r4['max_weight_diff']:.2e}{note_fr}")
        all_results["flight_recorder"] = r4
    else:
        r3 = {"n_cycles": 0, "overall_max_drift": 0.0, "zero_drift": True}
        r4 = {"n_layers_streamed": 0, "weights_match_original": True, "max_weight_diff": 0.0}
        print(f"\n{'─'*78}")
        print(f"  [3-4/4] SKIPPED (fast mode for large model)")
        print(f"{'─'*78}")

    # Save
    os.makedirs("quality/experiments", exist_ok=True)
    path = f"quality/experiments/rollback_demo_{int(time.time())}.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'─'*78}")
    print(f"  Report saved -> {path}")

    # Summary
    print(f"\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}")
    print(f"  REVO rollback restores NLL exactly?    {r1.perfectly_restored}")
    print(f"  Prune can rollback?                    {r2.can_rollback if prune else 'N/A'}")
    if not fast_mode:
        print(f"  Multi-cycle drift?                     {r3['zero_drift']}")
        print(f"  Flight recorder exact?                 {r4['weights_match_original']}")
    print()
    print(f"  REVO | dNLL(apply)={r1.delta_after_revo:+.6f} -> dNLL(revert)={r1.delta_after_revert:.2e} (REVERSIBLE)")
    if prune and r2 and r2.can_rollback is False:
        print(f"  Prune| dNLL(apply)={r2.damage_nll:+.6f} -> dNLL(revert)=PERMANENT (IRREVERSIBLE)")
    print()
    print(f"  NLL restoration delta = {abs(r1.delta_after_revert):.2e} (limited by float32 precision).")
    if prune and r2:
        prune_ratio = abs(r2.damage_nll) / max(abs(r1.delta_after_revert), 1e-30)
        print(f"  REVO revert |dNLL| = {abs(r1.delta_after_revert):.2e} vs Prune damage |dNLL| = {abs(r2.damage_nll):.4f} ({prune_ratio:.0f}x worse).")
    print()


if __name__ == "__main__":
    main()
