"""
REVO benchmark: baseline vs SVD+gradient-correction vs 4-bit quantization.

Usage:
    python examples/bench_phi3.py                           # distilgpt2 (default)
    python examples/bench_phi3.py --model Qwen/Qwen2.5-0.5B
    python examples/bench_phi3.py --model microsoft/phi-3-mini-4k-instruct

Output: quality/compare/bench_{model_short}.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from revo._utils import evaluate_nll, load_model_tokenizer, gen_texts

SEED = 42


def model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-", "_").lower()


def run(model_id: str, n_seqs: int, max_len: int, grad_steps: int) -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"Loading {model_id}...")
    model, tokenizer = load_model_tokenizer(model_id)
    n_params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    texts = gen_texts(n_seqs)
    results: dict = {
        "model": model_id, "params": n_params,
        "n_seqs": n_seqs, "max_len": max_len,
    }

    # ── Baseline ──────────────────────────────────────────────
    t0 = time.time()
    nll_base = evaluate_nll(model, tokenizer, texts, max_length=max_len)
    results["baseline"] = {"nll": nll_base, "time_s": time.time() - t0}
    print(f"\nBaseline NLL: {nll_base:.6f}")

    # ── SVD + gradient correction ────────────────────────────
    from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
    from revo.act_svd import safe_compress

    print(f"\n── SVD + gradient correction ──")
    t0 = time.time()
    prof = profile_model_2d(model)
    ranks = allocate_ranks_energy_with_caps(prof, energy_keep=0.95, max_rank=32)
    active = {k: v for k, v in ranks.items() if v > 0}
    print(f"  Layers profiled: {len(ranks)}, compressible: {len(active)}")

    result = safe_compress(
        model, ranks, texts, tokenizer,
        max_length=max_len,
        grad_steps=grad_steps, grad_lr=3e-5,
        target_delta_nll=0.5, device=device,
    )
    t_svd = time.time() - t0

    svd_code = result.copy()
    svd_code.pop("handles", None)
    svd_code.pop("_debug_errors", None)
    params_after = sum(p.numel() for p in model.parameters())
    ratio_actual = n_params / max(params_after, 1)
    results["svd_gradcorr"] = svd_code | {"compression_ratio": round(ratio_actual, 2), "time_s": t_svd}

    print(f"  SVD NLL:       {result['nll_svd']:.4f}  Δ={result['nll_svd'] - nll_base:+.4f}")
    print(f"  Corrected NLL: {result['nll_corrected']:.4f}  Δ={result['nll_corrected'] - nll_base:+.4f}")
    print(f"  Final NLL:     {result['nll_final']:.4f}  Δ={result['nll_delta_final']:+.4f}")
    print(f"  Compressed: {len(result['compressed'])}, Reverted: {len(result['reverted'])}")

    # ── 4-bit quantization ────────────────────────────────────
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n── Loading model for quantization ──")
    model, tokenizer = load_model_tokenizer(model_id)
    model.eval()

    from revo.gptq_revo import quantize_all, selective_dequantize

    print(f"── 4-bit quantization (group=32) ──")
    t0 = time.time()
    q_handles = quantize_all(
        model, texts, tokenizer,
        max_length=max_len,
        bits=4, group_size=32, device=device,
    )
    nll_4bit = evaluate_nll(model, tokenizer, texts, max_length=max_len)
    t_quant = time.time() - t0
    delta_4bit = nll_4bit - nll_base
    results["quant_4bit"] = {
        "nll": nll_4bit, "delta_nll": delta_4bit,
        "compression": "8.0×", "revertible": True,
        "bits": 4, "group_size": 32, "time_s": t_quant,
    }
    print(f"  Quant NLL: {nll_4bit:.4f}  Δ={delta_4bit:+.4f}")

    if delta_4bit > 0.5:
        def eval_fn():
            return evaluate_nll(model, tokenizer, texts, max_length=max_len)
        q_handles, reverted_q = selective_dequantize(
            model, q_handles, eval_fn, max_nll_delta=0.5,
        )
        nll_sq = evaluate_nll(model, tokenizer, texts, max_length=max_len)
        results["quant_4bit"]["nll_after_sq"] = nll_sq
        results["quant_4bit"]["modules_dequantized"] = len(reverted_q)
        print(f"  After SQ: {nll_sq:.4f}  Δ={nll_sq - nll_base:+.4f}")

    # ── Save ──────────────────────────────────────────────────
    out = Path(f"quality/compare/bench_{model_short_name(model_id)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Model':<25} {'Variant':<20} {'ΔNLL':<10} {'Compression':<10}")
    print(f"{'='*65}")
    print(f"{model_id:<25} {'baseline':<20} {'—':<10} {'1×':<10}")
    svd = results.get("svd_gradcorr", {})
    if svd:
        dn = svd.get("nll_delta_final", svd.get("nll_final", 0) - results.get("baseline", {}).get("nll", 0))
        cr = svd.get("compression_ratio", "?")
        print(f"{model_id:<25} {'SVD + grad corr':<20} {dn:+8.4f}  {cr}")
    q4 = results.get("quant_4bit", {})
    if q4:
        dn4 = q4.get("delta_nll", 0)
        print(f"{model_id:<25} {'4-bit (group=32)':<20} {dn4:+8.4f}  {q4.get('compression', '?'):<10}")
    print(f"{'='*65}")


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO benchmark: SVD + 4-bit")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--grad-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    run(args.model, args.prompts, args.max_length, args.grad_steps)


if __name__ == "__main__":
    main()
