"""REVO WeightLaw — the model IS the law.

Two paths, one revolution:

 Path A — DELTA MODE (fast, 1.3×, real weights in RAM):
   W_eff @ x = W_real @ x + U_eff @ (Vh @ x)
   Factored forward avoids reconstruction. 86K law params, 49ms/tok.

 Path B — PURE MODE (freed 2D weights, weights generated on-the-fly):
   y = U_eff @ (Vh @ x)
   All 2D block weights freed from RAM. The law IS the model.

Usage:
    python demo_revo_weightlaw.py          # Runs on SmolLM2-135M
    python demo_revo_weightlaw.py tiny     # Quick test on tiny-gpt2
"""

import sys, time, torch, torch.nn.functional as F
from revo.law_streaming import (
    build_law, _extract_svd_targets, pretrain_law,
    law_generate, law_stream_forward,
    _save_block_templates, _can_use_native,
)
from revo._utils import load_model_tokenizer, free_memory_trim, measure_memory_rss
from revo.streaming import _get_device, _free_all_block_2d_weights
from revo.streaming import _free_all_block_weights


def demo(model_name: str, rank: int, small_dim: int, hidden_dim: int,
         tokens: int, tiny: bool = False):

    print(f"Loading {model_name}...")
    model, tokenizer = load_model_tokenizer(model_name)
    device = _get_device(model)
    n_model = sum(p.numel() for p in model.parameters())

    # ─── Build law ───────────────────────────────────────────────────
    law = build_law(model, rank=rank, small_dim=small_dim,
                    hidden_dim=hidden_dim).to(device)
    n_law = sum(p.numel() for p in law.parameters())
    print(f"\n{'='*60}")
    print(f"  Law: {n_law:,} params = {n_law/n_model*100:.2f}% of model")
    print(f"  Rank: {rank}, small_dim: {small_dim}")
    print(f"  Delta mode (rank≤8): {law.rank <= 8}")
    print(f"{'='*60}")

    # ─── Quality check (delta: untrained; pure: SVD pretrained) ──────
    enc = tokenizer("The future of artificial intelligence",
                    return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits_full = model(enc).logits

    if law.rank <= 8:
        # Path A: delta — zero-shot, no training needed
        logits_law = law_stream_forward(model, enc, law, use_cache=False,
                                         use_native=True)
        label = "DELTA (zero-shot, untrained)"
    else:
        # Path B: pure — needs SVD pretraining
        targets = _extract_svd_targets(model, rank=rank)
        pretrain_law(law, targets, steps=50, device=device)
        logits_law = law_stream_forward(model, enc, law, use_cache=False,
                                         pure=True, use_native=True)
        label = "PURE (50-step SVD pretraining)"

    nll_full = F.cross_entropy(logits_full[:, :-1].reshape(-1, logits_full.shape[-1]),
                                enc[:, 1:].reshape(-1)).item()
    nll_law = F.cross_entropy(logits_law[:, :-1].reshape(-1, logits_law.shape[-1]),
                               enc[:, 1:].reshape(-1)).item()
    diff = (logits_full - logits_law).abs().max().item()

    print(f"\n  [{label}]")
    print(f"  NLL: full={nll_full:.4f}  law={nll_law:.4f}  Δ={nll_law-nll_full:+.4f}")
    print(f"  |logit|_∞ = {diff:.2e}", end="")
    if diff < 0.1:
        print("  ← identical to full model")
    else:
        print("  ← degraded (needs more training or higher rank)")

    # ─── Free 2D block weights (for pure mode) ──────────────────────
    if law.rank > 8:
        templates = _save_block_templates(model)
        rss_before = measure_memory_rss()
        _free_all_block_2d_weights(model)
        free_memory_trim()
        rss_after = measure_memory_rss()
        print(f"  Memory: {rss_before/1024/1024:.0f} → {rss_after/1024/1024:.0f} MB")
        block_templates = templates
    else:
        block_templates = None

    # ─── Benchmark speed ────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(enc).logits
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = law_stream_forward(model, enc, law, use_cache=False,
                            use_native=True, pure=(law.rank > 8))
    t_law = time.perf_counter() - t0

    print(f"  Speed: full={t_full*1000:.0f}ms  law={t_law*1000:.0f}ms  ({t_law/t_full:.1f}×)")

    # ─── Generate ──────────────────────────────────────────────────
    prompt = "The future of artificial intelligence"
    print(f"\n  Generating {tokens} tokens...")
    t0 = time.time()
    out_text, _ = law_generate(model, tokenizer, prompt, law,
                                max_new_tokens=tokens, temperature=0.7,
                                top_k=40, verbose=False,
                                pure=(law.rank > 8),
                                block_templates=block_templates)
    t_gen = time.time() - t0
    print(f"  {t_gen/tokens*1000:.0f}ms/tok  ({t_gen:.1f}s total)")
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {out_text[:250]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"

    if mode == "tiny":
        # Tiny test (10 seconds)
        demo("sshleifer/tiny-gpt2", rank=4, small_dim=2,
             hidden_dim=64, tokens=20, tiny=True)
    elif mode == "pure":
        # Pure WeightLaw (takes ~2 min for SVD pretraining)
        demo("HuggingFaceTB/SmolLM2-135M", rank=128, small_dim=32,
             hidden_dim=1024, tokens=40)
    else:
        # Fast delta mode — the REVOLUTION (no training)
        demo("HuggingFaceTB/SmolLM2-135M", rank=4, small_dim=2,
             hidden_dim=64, tokens=40)
