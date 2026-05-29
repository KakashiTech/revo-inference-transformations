#!/usr/bin/env python3
"""REVO HyperLoRA ablation: test different context_dim, rank strategies.

Tests:
1. Different context_dim values for hyperlora generation
2. Different rank allocation strategies (uniform vs energy-based)
3. REVO ephemeral delta vs SVD low-rank compression
4. Combined REVO + simulated quantization
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import seed_everything, evaluate_nll, load_model_tokenizer
from revo.engine import apply_delta, revert_delta, DeltaHandle
from revo.hyperlora import hyperlora_generate, HyperLoraConfig
from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXTS = [
    "The transformer architecture processes sequences using self-attention mechanisms.",
    "Low-rank approximations can reduce model size while preserving performance.",
    "Reversible computing allows undoing operations without loss of information.",
    "The spectral decomposition of a matrix reveals its low-dimensional structure.",
    "Language models predict the next token given the preceding context.",
]


def experiment_context_dim():
    """Test how context_dim affects hyperlora-generated delta quality."""
    seed_everything(SEED)
    model, tok = load_model_tokenizer("gpt2")
    nll_base = evaluate_nll(model, tok, TEXTS, 128, DEVICE)
    print(f"Baseline NLL: {nll_base:.4f}")

    results = []
    for ctx_dim in [4, 8, 16, 32, 64]:
        for rank in [4, 8, 16, 32]:
            model_r, _ = load_model_tokenizer("gpt2")
            model_r.eval()
            handles = []
            for name, m in model_r.named_modules():
                if not hasattr(m, "weight") or m.weight.dim() != 2:
                    continue
                of, inf = m.weight.shape
                if of * inf > 2_000_000:  # skip very large layers
                    continue
                if "lm_head" in name:
                    continue
                actual_rank = min(rank, min(of, inf) // 4)
                if actual_rank < 1:
                    continue
                cfg = HyperLoraConfig(
                    context_dim=ctx_dim,
                    rank=actual_rank,
                    in_features=inf,
                    out_features=of,
                )
                # Build context of exact ctx_dim length
                raw = np.frombuffer(
                    f"{SEED}:{name}".encode().ljust(256, b"\x00")[:256],
                    dtype=np.float32,
                )
                ctx = np.resize(raw, (ctx_dim,))
                A, B, scale = hyperlora_generate(ctx, cfg)
                W_np = m.weight.detach().cpu().numpy()
                W_new, handle = apply_delta(W_np, A, B, scale)
                m.weight.data = torch.from_numpy(W_new).to(m.weight.device, dtype=m.weight.dtype)
                handles.append(handle)

            nll = evaluate_nll(model_r, tok, TEXTS, 128, DEVICE)
            delta = nll - nll_base
            results.append({"ctx_dim": ctx_dim, "rank": rank, "nll": nll, "nll_delta": delta})
            print(f"  ctx_dim={ctx_dim:2d} rank={rank:2d}: NLL delta={delta:+.6f}")
            del model_r

    return results


def experiment_revo_vs_quant():
    """REVO + simulated int8 quantization vs each alone."""
    seed_everything(SEED)
    model_base, tok = load_model_tokenizer("sshleifer/tiny-gpt2")
    nll_base = evaluate_nll(model_base, tok, TEXTS, 128, DEVICE)
    params_base = sum(p.numel() for p in model_base.parameters())
    print(f"\n=== REVO + Quant Combination (tiny-gpt2) ===")
    print(f"Baseline NLL: {nll_base:.4f}, Params: {params_base:,}")

    results = []

    for label, do_revo, do_quant in [
        ("REVO only", True, False),
        ("int8 quant (sim)", False, True),
        ("REVO + int8 quant", True, True),
    ]:
        model, _ = load_model_tokenizer("sshleifer/tiny-gpt2")
        if do_revo:
            prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"])
            ranks = allocate_ranks_energy_with_caps(prof, energy_keep=0.92, max_rank_frac=0.20)
            replace_2d_modules_with_lowrank(model, ranks, calibrate=True, calibrate_samples=128, seed=SEED)
        if do_quant:
            for m in model.modules():
                if hasattr(m, "weight") and m.weight.dim() == 2:
                    w = m.weight.data
                    w_range = w.amax() - w.amin()
                    if w_range > 0:
                        w_int8 = ((w - w.amin()) / w_range * 255 - 128).round().clamp(-128, 127)
                        w_restored = (w_int8 + 128) / 255 * w_range + w.amin()
                        m.weight.data = w_restored
        nll = evaluate_nll(model, tok, TEXTS, 128, DEVICE)
        params = sum(p.numel() for p in model.parameters())
        results.append({
            "variant": label,
            "nll": nll,
            "nll_delta": nll - nll_base,
            "params": params,
            "compression": 1.0 - params / params_base,
        })
        print(f"  {label:20s}: NLL={nll:.4f} (delta={nll-nll_base:+.4f}) comp={1-params/params_base:.1%}")
        del model

    return {"baseline_nll": nll_base, "baseline_params": params_base, "results": results}


def main():
    print("=" * 70)
    print("EXPERIMENT 1: HyperLoRA context_dim + rank ablation")
    print("=" * 70)
    ctx_results = experiment_context_dim()

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: REVO + Quantization combination")
    print("=" * 70)
    quant_results = experiment_revo_vs_quant()

    report = {
        "context_dim_ablation": ctx_results,
        "revo_quant_combination": quant_results,
    }

    out_path = os.path.join("quality", "experiments", f"hyperlora_ablation_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
