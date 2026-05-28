#!/usr/bin/env python3
"""REVO Fusion: activation-aware SVD + module replacement + reversible tails.

Compares:
  1. Naive SVD (replace_2d_modules_with_lowrank) — compression, no revert
  2. Act-aware SVD + module replacement + tail handles — compression + revert
  3. Baseline (no compression)

Measures NLL delta, compression ratio, revert accuracy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import encode_text, evaluate_nll, seed_everything
from revo.act_svd import (
    replace_with_act_svd_compression,
    revert_act_svd_compression,
)

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CALIB_TEXTS = [
    "The singular value decomposition reveals the intrinsic dimensionality of linear transformations.",
    "Activation-aware compression minimizes the output error rather than the weight error.",
    "Low-rank approximations enable efficient inference while preserving model quality.",
    "The residual connection in transformers allows gradients to flow through many layers.",
    "Matrix factorization techniques decompose large weight matrices into smaller components.",
]
MAX_LEN = 64


@dataclass
class FusionResult:
    method: str
    energy_keep: float
    nll_baseline: float
    nll_compressed: float
    nll_delta: float
    nll_after_revert: float
    revert_delta: float
    perfectly_restored: bool
    params_before: int
    params_after: int
    compression_ratio: float
    n_modules: int
    time_s: float
    model: str


def count_weight_params(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
            total += m.weight.numel()
    return total


def eval_nll(model, tok, texts):
    return evaluate_nll(model, tok, texts, max_length=MAX_LEN, device=DEVICE)


def get_ranks(model, ek: float) -> Dict[str, int]:
    from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
    prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], max_rank=None)
    return allocate_ranks_energy_with_caps(prof, energy_keep=ek, max_rank=None, max_rank_frac=0.25)


def get_texts():
    if load_dataset is not None:
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts = [ex["text"] for ex in ds if ex["text"].strip()]
            return texts[:20]
        except Exception:
            pass
    return CALIB_TEXTS


def run_test(model_name: str, method: str, ek: float, shared_ranks: Optional[Dict[str, int]] = None) -> FusionResult:
    seed_everything(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = get_texts()

    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()
    nll_base = evaluate_nll(model, tok, texts, max_length=MAX_LEN, device=DEVICE)
    params_before = count_weight_params(model)

    if shared_ranks is not None:
        ranks = shared_ranks
    else:
        prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"], max_rank=None)
        ranks = allocate_ranks_energy_with_caps(prof, energy_keep=ek, max_rank=None, max_rank_frac=0.25)

    t0 = time.perf_counter()

    if method == "naive_svd":
        from revo.lowrank import replace_2d_modules_with_lowrank
        report = replace_2d_modules_with_lowrank(
            model, ranks, calibrate=True, calibrate_samples=256,
            seed=SEED, fold_calibration=True,
        )
        dt = time.perf_counter() - t0
        nll_comp = eval_nll(model, tok, texts)
        params_after = count_weight_params(model)
        return FusionResult(
            method=method, energy_keep=ek,
            nll_baseline=nll_base, nll_compressed=nll_comp,
            nll_delta=nll_comp - nll_base,
            nll_after_revert=float("inf"), revert_delta=float("inf"),
            perfectly_restored=False,
            params_before=params_before, params_after=params_after,
            compression_ratio=params_before / max(params_after, 1),
            n_modules=len(report), time_s=dt, model=model_name,
        )

    elif method == "act_svd_nocov":
        handles = replace_with_act_svd_compression(
            model, ranks, calibrate=False,
        )
        dt = time.perf_counter() - t0
        nll_comp = eval_nll(model, tok, texts)
        params_after = count_weight_params(model)
        revert_act_svd_compression(model, handles)
        nll_revert = eval_nll(model, tok, texts)
        return FusionResult(
            method=method, energy_keep=ek,
            nll_baseline=nll_base, nll_compressed=nll_comp,
            nll_delta=nll_comp - nll_base,
            nll_after_revert=nll_revert, revert_delta=abs(nll_revert - nll_base),
            perfectly_restored=abs(nll_revert - nll_base) < 1e-8,
            params_before=params_before, params_after=params_after,
            compression_ratio=params_before / max(params_after, 1),
            n_modules=len(handles), time_s=dt, model=model_name,
        )

    elif method == "act_svd":
        handles = replace_with_act_svd_compression(
            model, ranks,
        )
        dt = time.perf_counter() - t0
        nll_comp = eval_nll(model, tok, texts)
        params_after = count_weight_params(model)
        revert_act_svd_compression(model, handles)
        nll_revert = eval_nll(model, tok, texts)
        return FusionResult(
            method=method, energy_keep=ek,
            nll_baseline=nll_base, nll_compressed=nll_comp,
            nll_delta=nll_comp - nll_base,
            nll_after_revert=nll_revert, revert_delta=abs(nll_revert - nll_base),
            perfectly_restored=abs(nll_revert - nll_base) < 1e-8,
            params_before=params_before, params_after=params_after,
            compression_ratio=params_before / max(params_after, 1),
            n_modules=len(handles), time_s=dt, model=model_name,
        )

    raise ValueError(f"Unknown method: {method}")


def main():
    models = ["sshleifer/tiny-gpt2"]
    try:
        AutoModelForCausalLM.from_pretrained("gpt2")
        models.append("gpt2")
    except Exception:
        print("[gpt2 not available, skipping]")

    eks = [0.99, 0.95, 0.92, 0.85]
    methods = ["naive_svd", "act_svd"]

    print("=" * 80)
    print("REVO FUSION: Activation-aware SVD + Module Replacement + Reversible Tails")
    print("=" * 80)

    all_results: List[Dict[str, Any]] = []

    for model_name in models:
        print(f"\n{'='*80}")
        print(f" Model: {model_name}")
        print(f"{'='*80}")

        for ek in eks:
            print(f"\n  Energy keep: {ek}")
            # Compute ranks once per model+ek to ensure identical compression
            model_tmp = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
            model_tmp.eval()
            shared_ranks = get_ranks(model_tmp, ek)
            del model_tmp

            for method in methods:
                try:
                    r = run_test(model_name, method, ek, shared_ranks=shared_ranks)
                    label = method.ljust(15)
                    delta = r.nll_delta
                    ratio = r.compression_ratio
                    rev = r.revert_delta
                    perfect = r.perfectly_restored
                    print(f"    {label} ΔNLL={delta:+.4f}  ratio={ratio:.2f}x  "
                          f"revert={rev:.2e}  perfect={perfect}")
                    all_results.append(asdict(r))
                except Exception as e:
                    print(f"    {method:<15} FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary table
    print(f"\n{'='*80}")
    print(" SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<18} {'Energy':<8} {'ΔNLL':<10} {'Ratio':<8} {'Revert':<12} {'Perfect?':<8}")
    print("-" * 75)
    for r in all_results:
        d_str = f"{r['nll_delta']:+.4f}" if r['nll_delta'] != float("inf") else "FAIL"
        rd_str = f"{r['revert_delta']:.2e}" if r['revert_delta'] != float("inf") else "N/A"
        perf = "YES" if r['perfectly_restored'] else "NO"
        print(f"{r['method']:<18} {r['energy_keep']:<8.2f} {d_str:<10} "
              f"{r['compression_ratio']:<8.2f}x {rd_str:<12} {perf:<8}")

    os.makedirs("quality/experiments", exist_ok=True)
    path = "quality/experiments/revo_fusion_benchmark.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {path}")


if __name__ == "__main__":
    main()
