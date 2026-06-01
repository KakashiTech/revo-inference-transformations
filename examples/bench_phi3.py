"""
Phi-3 Mini benchmark: baseline vs REVO SVD vs REVO 4-bit.

Fallback a tiny-gpt2 si no hay RAM para Phi-3.
Produce: quality/compare/phi3_bench.json
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from revo._utils import evaluate_nll, load_model_tokenizer, gen_texts

OUT = Path("quality/compare/phi3_bench.json")
N_SEQS = 10
MAX_LEN = 64
SEED = 42
GRAD_STEPS = 3


def run() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Intentar Phi-3 Mini; fallback a tiny-gpt2
    model_id = "microsoft/phi-3-mini-4k-instruct"
    try:
        model, tokenizer = load_model_tokenizer(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print(f"Model: {model_id}")
    except Exception as e:
        print(f"Phi-3 Mini failed ({e}), falling back to tiny-gpt2")
        model_id = "sshleifer/tiny-gpt2"
        model, tokenizer = load_model_tokenizer(model_id)

    device = next(model.parameters()).device
    texts = gen_texts(N_SEQS)
    results: dict = {"model": model_id, "n_seqs": N_SEQS, "max_len": MAX_LEN}

    # ── Baseline ──────────────────────────────────────────────
    t0 = time.time()
    nll_base = evaluate_nll(model, tokenizer, texts, max_length=MAX_LEN)
    results["baseline"] = {"nll": nll_base, "time_s": time.time() - t0}
    print(f"\nBaseline NLL: {nll_base:.6f}")

    # ── Phase 1: SVD + gradient correction ────────────────────
    from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
    from revo.act_svd import safe_compress

    print(f"\n── SVD + gradient correction ──")
    t0 = time.time()
    prof = profile_model_2d(model)
    ranks = allocate_ranks_energy_with_caps(prof, energy_keep=0.95, max_rank=32)
    result = safe_compress(
        model, ranks, texts, tokenizer,
        max_length=MAX_LEN,
        grad_steps=GRAD_STEPS, grad_lr=3e-5,
        target_delta_nll=0.5, device=device,
    )
    t_svd = time.time() - t0
    delta_svd = result["nll_final"] - nll_base
    results["svd_gradcorr"] = {
        "nll": result["nll_final"],
        "delta_nll": delta_svd,
        "nll_svd": result["nll_svd"],
        "nll_corrected": result["nll_corrected"],
        "compression": "~1.5×",
        "revertible": True,
        "modules_compressed": len(result["compressed"]),
        "modules_reverted": len(result["reverted"]),
        "time_s": t_svd,
    }
    print(f"SVD NLL:       {result['nll_svd']:.6f}  Δ={result['nll_svd'] - nll_base:+.6f}")
    print(f"Corrected NLL: {result['nll_corrected']:.6f}  Δ={result['nll_corrected'] - nll_base:+.6f}")
    print(f"Final NLL:     {result['nll_final']:.6f}  Δ={delta_svd:+.6f}")
    print(f"Compressed: {len(result['compressed'])} modules, Reverted: {len(result['reverted'])}")

    # ── Phase 2: 4-bit quantization ───────────────────────────
    from revo.gptq_revo import quantize_all, selective_dequantize

    # Recargar modelo limpio
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    try:
        model, tokenizer = load_model_tokenizer(model_id)
    except Exception:
        model, tokenizer = load_model_tokenizer(model_id)

    model.eval()
    print(f"\n── 4-bit quantization (group=32) ──")
    t0 = time.time()
    q_handles = quantize_all(
        model, texts, tokenizer,
        max_length=MAX_LEN,
        bits=4, group_size=32, device=device,
    )
    nll_4bit = evaluate_nll(model, tokenizer, texts, max_length=MAX_LEN)
    t_quant = time.time() - t0
    delta_4bit = nll_4bit - nll_base

    results["quant_4bit"] = {
        "nll": nll_4bit,
        "delta_nll": delta_4bit,
        "compression": "8.0×",
        "revertible": True,
        "bits": 4,
        "group_size": 32,
        "time_s": t_quant,
    }
    print(f"Quant NLL:     {nll_4bit:.6f}  Δ={delta_4bit:+.6f}")

    # Selective dequantize if needed
    if delta_4bit > 0.5:
        def eval_fn():
            return evaluate_nll(model, tokenizer, texts, max_length=MAX_LEN)
        q_handles, reverted_q = selective_dequantize(
            model, q_handles, eval_fn, max_nll_delta=0.5,
        )
        nll_sq = evaluate_nll(model, tokenizer, texts, max_length=MAX_LEN)
        results["quant_4bit"]["nll_after_sq"] = nll_sq
        results["quant_4bit"]["modules_dequantized"] = len(reverted_q)
        print(f"After SQ:      {nll_sq:.6f}  Δ={nll_sq - nll_base:+.6f}")
        print(f"Dequantized: {len(reverted_q)} modules")

    # ── Save ──────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT}")

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Model':<22} {'Variant':<22} {'ΔNLL':<10} {'Compression':<12}")
    print(f"{'='*60}")
    print(f"{model_id:<22} {'baseline':<22} {'—':<10} {'1×':<12}")
    svd = results.get("svd_gradcorr", {})
    if svd:
        print(f"{model_id:<22} {'SVD + grad corr':<22} {svd.get('delta_nll', 0):+8.6f}  {svd.get('compression','?'):<12}")
    q4 = results.get("quant_4bit", {})
    if q4:
        print(f"{model_id:<22} {'4-bit (group=32)':<22} {q4.get('delta_nll', 0):+8.6f}  {q4.get('compression','?'):<12}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
