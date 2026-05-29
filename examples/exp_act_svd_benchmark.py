#!/usr/bin/env python3
"""Compare naive SVD vs Activation-aware SVD for REVO compression.

Measures:
  1. NLL / Perplexity at multiple compression ratios
  2. Reversibility (can we revert exactly?)
  3. Compression ratio vs quality tradeoff

Runs on tiny-gpt2 (fast) + GPT-2 (real scale).
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
from revo.lowrank import compress_linear_to_lowrank, LowRankLinear
from revo.act_svd import (
    apply_activation_aware_svd_to_model,
    revert_svd_compression,
    SVDTailHandle,
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
class CompressionResult:
    method: str
    energy_keep: float
    rank_frac: float
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_params_of_weight(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
            total += m.weight.numel()
    return total


def get_weight_ranks(model, energy_keep: float, max_rank_frac: float = 0.25):
    """Get rank allocation based on SVD energy profile."""
    from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
    prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"],
                            max_rank=None)
    ranks = allocate_ranks_energy_with_caps(prof, energy_keep=energy_keep,
                                            max_rank=None,
                                            max_rank_frac=max_rank_frac)
    return prof, ranks


def eval_ppl(model, tok, texts, max_len=MAX_LEN):
    """Evaluate perplexity = exp(NLL)."""
    nll = evaluate_nll(model, tok, texts, max_length=max_len, device=DEVICE)
    return nll, float(torch.exp(torch.tensor(nll)))


def get_wikitext2_texts(split: str = "test", n_samples: int = 20):
    """Get WikiText-2 texts for evaluation."""
    if load_dataset is None:
        return None
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split,
                          trust_remote_code=True)
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
        return texts[:n_samples]
    except Exception as e:
        print(f"  [WikiText-2 not available: {e}]")
        return None


def run_naive_svd_experiment(model_name: str, energy_keep: float,
                             max_rank_frac: float,
                             calib_texts: List[str]) -> CompressionResult:
    """Naive SVD via compress_linear_to_lowrank (full replacement)."""
    seed_everything(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    # Compute energy-keep profile ranks (from the SVD profile BEFORE compression)
    from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
    prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"],
                            max_rank=None)
    ranks = allocate_ranks_energy_with_caps(prof, energy_keep=energy_keep,
                                            max_rank=None,
                                            max_rank_frac=max_rank_frac)

    nll_base, ppl_base = eval_ppl(model, tok, calib_texts)
    params_before = count_params_of_weight(model)

    # Apply compression (replace modules with LowRankLinear)
    from revo.lowrank import replace_2d_modules_with_lowrank
    t0 = time.perf_counter()
    report = replace_2d_modules_with_lowrank(
        model, ranks, calibrate=True, calibrate_samples=256, seed=SEED,
        fold_calibration=True,
    )
    dt = time.perf_counter() - t0

    nll_comp, ppl_comp = eval_ppl(model, tok, calib_texts)
    params_after = count_params(model)

    # Naive SVD is NOT reversible (modules replaced, no tail stored)
    # The revert delta is ∞ since there's no revert capability
    return CompressionResult(
        method="naive_svd",
        energy_keep=energy_keep, rank_frac=max_rank_frac,
        nll_baseline=nll_base,
        nll_compressed=nll_comp,
        nll_delta=nll_comp - nll_base,
        nll_after_revert=float("inf"),
        revert_delta=float("inf"),
        perfectly_restored=False,
        params_before=params_before,
        params_after=params_after,
        compression_ratio=params_before / max(params_after, 1),
        n_modules=len(report),
        time_s=dt,
        model=model_name,
    )


def run_activation_aware_svd(model_name: str, energy_keep: float,
                              max_rank_frac: float,
                              calib_texts: List[str],
                              use_covariance: bool = True) -> CompressionResult:
    """Activation-aware SVD compression (reversible via tail handles)."""
    seed_everything(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
    prof = profile_model_2d(model, name_patterns=["attn", "mlp", "c_fc", "c_proj"],
                            max_rank=None)
    ranks = allocate_ranks_energy_with_caps(prof, energy_keep=energy_keep,
                                            max_rank=None,
                                            max_rank_frac=max_rank_frac)

    nll_base, ppl_base = eval_ppl(model, tok, calib_texts)
    params_before = count_params_of_weight(model)

    t0 = time.perf_counter()
    handles = apply_activation_aware_svd_to_model(
        model, ranks,
        calibrate=use_covariance,
        calibrate_texts=calib_texts if use_covariance else None,
        tokenizer=tok,
        max_length=MAX_LEN,
        seed=SEED,
        device=DEVICE,
    )
    dt = time.perf_counter() - t0

    nll_comp, ppl_comp = eval_ppl(model, tok, calib_texts)
    params_after = count_params_of_weight(model)

    # Revert
    revert_svd_compression(model, handles)
    nll_revert, ppl_revert = eval_ppl(model, tok, calib_texts)
    revert_err = abs(nll_revert - nll_base)

    method_name = "act_svd" if use_covariance else "act_svd_nocov"
    return CompressionResult(
        method=method_name,
        energy_keep=energy_keep, rank_frac=max_rank_frac,
        nll_baseline=nll_base,
        nll_compressed=nll_comp,
        nll_delta=nll_comp - nll_base,
        nll_after_revert=nll_revert,
        revert_delta=revert_err,
        perfectly_restored=revert_err < 1e-6,
        params_before=params_before,
        params_after=params_after,
        compression_ratio=params_before / max(params_after, 1),
        n_modules=len(handles),
        time_s=dt,
        model=model_name,
    )


def main():
    models = ["sshleifer/tiny-gpt2"]
    try:
        _ = AutoModelForCausalLM.from_pretrained("gpt2")
        models.append("gpt2")
    except Exception:
        print("[gpt2 not available, skipping]")

    energy_configs = [
        (0.99, 0.30, "very high"),
        (0.95, 0.25, "high"),
        (0.92, 0.20, "medium"),
        (0.85, 0.15, "low"),
    ]

    print("=" * 80)
    print("ACTIVATION-AWARE SVD vs NAIVE SVD — REVO Compression Benchmark")
    print("=" * 80)

    all_results: List[Dict[str, Any]] = []

    for model_name in models:
        print(f"\n{'='*80}")
        print(f" Model: {model_name}")
        print(f"{'='*80}")

        # Use WikiText-2 if available, else CALIB_TEXTS
        eval_texts = get_wikitext2_texts(n_samples=20) or CALIB_TEXTS
        print(f"  Evaluation texts: {len(eval_texts)}")

        for energy_keep, rank_frac, label in energy_configs:
            print(f"\n  --- Energy keep: {energy_keep} ({label}) ---")

            # 1. Naive SVD
            try:
                r_naive = run_naive_svd_experiment(
                    model_name, energy_keep, rank_frac, eval_texts)
                print(f"  Naive SVD:     ΔNLL={r_naive.nll_delta:+.4f}  "
                      f"ratio={r_naive.compression_ratio:.2f}x  "
                      f"n_mod={r_naive.n_modules}  "
                      f"reversible={'NO' if r_naive.perfectly_restored is False else 'YES'}")
                all_results.append(asdict(r_naive))
            except Exception as e:
                print(f"  Naive SVD FAILED: {e}")
                import traceback
                traceback.print_exc()

            # 2. Activation-aware SVD (with covariance)
            try:
                r_act = run_activation_aware_svd(
                    model_name, energy_keep, rank_frac, eval_texts,
                    use_covariance=True)
                print(f"  Act-Aware SVD: ΔNLL={r_act.nll_delta:+.4f}  "
                      f"ratio={r_act.compression_ratio:.2f}x  "
                      f"n_mod={r_act.n_modules}  "
                      f"revert={r_act.revert_delta:.2e}  "
                      f"perfect={'YES' if r_act.perfectly_restored else 'NO'}")
                all_results.append(asdict(r_act))
            except Exception as e:
                print(f"  Act-Aware SVD FAILED: {e}")
                import traceback
                traceback.print_exc()

            # 3. Activation-aware SVD (without covariance — baseline comparison)
            try:
                r_act2 = run_activation_aware_svd(
                    model_name, energy_keep, rank_frac, eval_texts,
                    use_covariance=False)
                print(f"  Act-Aware (no cov): ΔNLL={r_act2.nll_delta:+.4f}  "
                      f"ratio={r_act2.compression_ratio:.2f}x  "
                      f"revert={r_act2.revert_delta:.2e}")
                all_results.append(asdict(r_act2))
            except Exception as e:
                print(f"  Act-Aware (no cov) FAILED: {e}")

    # Summary table
    print(f"\n{'='*80}")
    print(" SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Energy':<8} {'ΔNLL':<10} {'Ratio':<8} {'Revert':<12} {'Perfect?':<8}")
    print("-" * 75)
    for r in all_results:
        d = r["nll_delta"]
        d_str = f"{d:+.4f}" if d != float("inf") else "FAIL"
        rd = r["revert_delta"]
        rd_str = f"{rd:.2e}" if rd != float("inf") else "N/A"
        print(f"{r['method']:<25} {r['energy_keep']:<8.2f} {d_str:<10} "
              f"{r['compression_ratio']:<8.2f}x {rd_str:<12} "
              f"{'YES' if r.get('perfectly_restored') else 'NO':<8}")

    os.makedirs("quality/experiments", exist_ok=True)
    path = "quality/experiments/act_svd_benchmark.json"
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results: {path}")


if __name__ == "__main__":
    main()
