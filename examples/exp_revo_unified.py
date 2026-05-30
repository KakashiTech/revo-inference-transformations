#!/usr/bin/env python3
"""REVO Unified Pipeline — todos los módulos conectados al EphemeralEngine.

Demostración: crea un transformer pequeño, activa TODOS los weight-replacement
modules y runtime enhancements, y verifica que el pipeline corre end-to-end.
"""

from __future__ import annotations

import json
import time
import os
import torch
import torch.nn as nn

from revo.ephemeral_engine import EphemeralConfig, EphemeralEngine


class FakeBlock(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
        self.mlp = nn.Linear(d_model, d_model)


class TinyTransformer(nn.Module):
    def __init__(self, n_layers: int = 6, d_model: int = 64, vocab: int = 128):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([FakeBlock(d_model) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab)


def run_unified_experiment():
    print("=" * 60)
    print("REVO Unified Pipeline — todos los módulos conectados")
    print("=" * 60)

    model = TinyTransformer(n_layers=6, d_model=64, vocab=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo base: {n_params:,} parámetros")

    # Config activando TODO
    cfg = EphemeralConfig(
        # Weight replacements (modelo)
        use_spectral_pruning=True, spectral_energy_keep=0.9,
        use_wdm=True, wdm_bands=2,
        use_circulant=True,
        use_holography=True, holography_boundary_dim=8,
        use_phase_bus=True,
        use_reversible=True, reversible_rank=2,
        use_holomorphic=True, holomorphic_rank=4,
        use_hora=True, hora_rank=2,
        # Runtime enhancements
        use_beds=True, beds_h_min=0.5, beds_h_max=3.5,
        use_tqft=True, tqft_topk=64, tqft_num_braids=5,
        use_oscillatory=True, osc_alpha=0.05, osc_freq=0.25,
    )

    t0 = time.perf_counter()
    engine = EphemeralEngine(model, cfg)
    init_time = time.perf_counter() - t0

    print(f"\nInicialización: {init_time*1000:.1f} ms")
    print(f"Reemplazos aplicados: {list(engine._replacements_report.keys())}")

    for name, report in engine._replacements_report.items():
        n_items = len(report) if isinstance(report, dict) else report
        print(f"  • {name}: {n_items} módulos/tensores afectados")

    # Step simple
    h = torch.randn(1, 64)
    t0 = time.perf_counter()
    logits, meta = engine.step(h, 0)
    step_time = time.perf_counter() - t0

    print(f"\nSingle step: {step_time*1000:.1f} ms")
    print(f"  logits shape: {logits.shape}")
    print(f"  skip: {meta.get('skipped')}")
    if not meta.get('skipped'):
        print(f"  scale: {meta['final_scale']:.4f}")
        print(f"  cache: {meta.get('cache', 'N/A')}")

    # Multi-step sequence
    n_steps = 10
    t0 = time.perf_counter()
    for i in range(n_steps):
        h = torch.randn(1, 64)
        engine.step(h, i)
    seq_time = time.perf_counter() - t0

    stats = engine.stats()
    print(f"\n{n_steps} steps: {seq_time*1000:.1f} ms (avg {seq_time*1000/n_steps:.1f} ms/step)")
    print(f"  cache hits: {stats['cache_hits']}, misses: {stats['cache_misses']}")
    print(f"  sim hits: {stats['cache_similar_hits']}")
    print(f"  gated skip: {stats['gated_skip']}, reduced: {stats['gated_reduced']}")
    print(f"  total NLL delta: {stats['total_nll_delta']:.4f}")

    # Resumen
    wt = stats.get('weight_replacements', {})
    print(f"\n  weight_replacements en stats: {len(wt)} módulos")
    print(f"  campos en stats: {[k for k in stats.keys() if not isinstance(stats.get(k), dict)]}")

    print("\n" + "=" * 60)
    print("TODO el pipeline REVO funciona end-to-end ✅")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    stats = run_unified_experiment()
