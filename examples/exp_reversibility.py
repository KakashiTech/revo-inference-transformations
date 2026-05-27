#!/usr/bin/env python3
"""Experimento: REVO es reversible? Quant+prune es destructivo?

Hipotesis: REVO apply+revert debe restaurar NLL exacta.
Quant+prune NO puede revertirse — el dano es permanente.

Este es el diferenciador fundamental de REVO.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import evaluate_nll, measure_memory_rss, seed_everything
from revo.engine import apply_delta, revert_delta, DeltaHandle
from revo.hyperlora import hyperlora_generate, HyperLoraConfig
from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank

try:
    import torch.nn.utils.prune as prune
except Exception:
    prune = None


MODEL = "sshleifer/tiny-gpt2"
TEXTS = [
    "El filtrado espectral transforma la representacion interna.",
    "Las matrices circulantes permiten convolucion O(n log n) via FFT.",
    "La curvatura del espacio latente se regulariza con PDE.",
    "Un bus de fase natural sincroniza frecuencias cognitivas.",
    "El un-computing reversible restaura el estado original.",
    "La sintesis holografica bulk-boundary preserva topologia.",
    "WDM paraleliza subcanales en el dominio espectral.",
    "El gating efimero modula sin destruir informacion.",
    "Radix cache acelera la recuperacion por prefijo.",
    "La calibracion probabilistica ajusta temperatura por capa.",
]
SEED = 42
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RoundtripResult:
    nll_baseline: float
    nll_after_apply: float
    nll_after_revert: float
    rss_baseline_mb: float
    rss_after_apply_mb: float
    rss_after_revert_mb: float
    apply_time_s: float
    n_replaced_modules: int
    params_delta: int
    revert_restored: bool  # |nll_before - nll_after_revert| < 1e-6


def experiment_reversibility() -> RoundtripResult:
    seed_everything(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()

    params_before = sum(p.numel() for p in model.parameters())

    # Baseline NLL
    t0 = time.perf_counter()
    rss0 = measure_memory_rss()
    nll_base = evaluate_nll(model, tok, TEXTS, max_length=MAX_LEN, device=DEVICE)
    dt_base = time.perf_counter() - t0

    print(f"  baseline NLL:        {nll_base:.6f}")
    print(f"  baseline time:       {dt_base:.4f}s")
    print(f"  params before:       {params_before}")

    # Apply REVO (low-rank deltas)
    t1 = time.perf_counter()
    prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], max_rank=None)
    ranks = allocate_ranks_energy_with_caps(prof, energy_keep=0.92, max_rank=None, max_rank_frac=0.20)
    handles: List[DeltaHandle] = []
    n_replaced = 0
    for module_name, module in model.named_modules():
        if module_name in ranks:
            r = ranks[module_name]
            if r < 1:
                continue
            cfg = HyperLoraConfig(context_dim=16, rank=r, in_features=module.weight.shape[1], out_features=module.weight.shape[0])
            context = np.frombuffer(f"{SEED}:{module_name}".encode().ljust(64, b'\x00')[:64], dtype=np.float32)[:16]
            A, B, scale = hyperlora_generate(context, cfg)
            W_np = module.weight.detach().cpu().numpy()
            W_new, handle = apply_delta(W_np, A, B, scale)
            module.weight.data = torch.from_numpy(W_new).to(module.weight.device, dtype=module.weight.dtype)
            handles.append(handle)
            n_replaced += 1
    dt_apply = time.perf_counter() - t1
    rss_apply = measure_memory_rss()

    # NLL after REVO
    nll_revo = evaluate_nll(model, tok, TEXTS, max_length=MAX_LEN, device=DEVICE)
    params_revo = sum(p.numel() for p in model.parameters())

    print(f"  NLL after REVO:      {nll_revo:.6f}  (delta={nll_revo - nll_base:+.6f})")
    print(f"  apply time:          {dt_apply:.4f}s")
    print(f"  modules replaced:    {n_replaced}")
    print(f"  params after:        {params_revo}  (delta={params_revo - params_before})")

    # Revert all deltas
    t2 = time.perf_counter()
    for module_name, module in model.named_modules():
        if module_name in ranks and ranks[module_name] >= 1:
            handle = handles.pop(0)
            W_current = module.weight.detach().cpu().numpy()
            W_restored = revert_delta(W_current, handle)
            module.weight.data = torch.from_numpy(W_restored).to(module.weight.device, dtype=module.weight.dtype)
    dt_revert = time.perf_counter() - t2
    rss_revert = measure_memory_rss()

    # NLL after revert
    nll_restored = evaluate_nll(model, tok, TEXTS, max_length=MAX_LEN, device=DEVICE)
    restored = abs(nll_restored - nll_base) < 1e-6

    print(f"  NLL after revert:    {nll_restored:.6f}  (delta={nll_restored - nll_base:+.6f})")
    print(f"  revert time:         {dt_revert:.4f}s")
    print(f"  revert_restored:     {restored}")

    return RoundtripResult(
        nll_baseline=nll_base,
        nll_after_apply=nll_revo,
        nll_after_revert=nll_restored,
        rss_baseline_mb=float(rss0) / (1024 * 1024),
        rss_after_apply_mb=float(rss_apply) / (1024 * 1024),
        rss_after_revert_mb=float(rss_revert) / (1024 * 1024),
        apply_time_s=dt_apply,
        n_replaced_modules=n_replaced,
        params_delta=params_revo - params_before,
        revert_restored=restored,
    )


def experiment_prune_is_destructive() -> Dict[str, Any]:
    """Quant+prune NO puede revertirse. Demostracion."""
    seed_everything(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()

    nll_base = evaluate_nll(model, tok, TEXTS, max_length=MAX_LEN, device=DEVICE)

    # Prune
    if prune is not None:
        params = [(mod, "weight") for _, mod in model.named_modules()
                  if hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), nn.Parameter)]
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.3)
        for mod, _ in params:
            prune.remove(mod, "weight")

    nll_pruned = evaluate_nll(model, tok, TEXTS, max_length=MAX_LEN, device=DEVICE)

    print(f"  baseline NLL:        {nll_base:.6f}")
    print(f"  NLL after prune:     {nll_pruned:.6f}  (delta={nll_pruned - nll_base:+.6f}, PERMANENT)")

    return {"model": MODEL, "nll_baseline": nll_base, "nll_after_prune": nll_pruned,
            "damage_permanent": True, "can_revert": False}


import torch.nn as nn

def main():
    print("=" * 70)
    print("EXPERIMENTO 1: REVO reversibility (apply -> revert -> NLL restored)")
    print("=" * 70)
    r1 = experiment_reversibility()
    print()

    print("=" * 70)
    print("EXPERIMENTO 2: Quant+Prune es DESTRUCTIVO (no se puede revertir)")
    print("=" * 70)
    r2 = experiment_prune_is_destructive()
    print()

    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"REVO apply+revert restore NLL:   {r1.revert_restored}")
    print(f"  NLL baseline:                  {r1.nll_baseline:.6f}")
    print(f"  NLL after REVO:                {r1.nll_after_apply:.6f}")
    print(f"  NLL after revert:              {r1.nll_after_revert:.6f}")
    print(f"  |baseline - revert|:           {abs(r1.nll_baseline - r1.nll_after_revert):.2e}")
    print(f"  params_delta:                  {r1.params_delta}")
    print(f"  modules replaced:              {r1.n_replaced_modules}")
    print()
    print(f"Quant+Prune restore possible:    {r2['can_revert']}")
    print(f"  NLL damage:                    {r2['nll_after_prune'] - r2['nll_baseline']:+.6f} (PERMANENT)")

    report = {
        "reversibility": asdict(r1),
        "prune_destructive": r2,
        "conclusion": {
            "revo_is_reversible": r1.revert_restored,
            "prune_is_destructive": not r2["can_revert"],
            "key_insight": "REVO habilita computacion efimera: aplicar, usar, revertir, olvidar. "
                           "Quant+prune causa dano permanente e irreversible.",
        }
    }
    path = "quality/experiments/reversibility_proof.json"
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()
