"""Context encoder comparison: heuristic vs window(4) — the key test."""

from __future__ import annotations

import json, time, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from revo._logging import get_logger
from revo.context_encoder import WindowContextEncoder
from revo.engine import apply_delta, revert_delta
from revo.hyperlora import HyperLoraConfig, hyperlora_generate

log = get_logger(__name__)

d_model = 768
context_dim = 16
rank = 4

# Longer, more varied prompts for better statistics
LONG_TEXTS = [
    "The capital of France is Paris and it has been for centuries a center of art and culture",
    "Quantum computing relies on superposition and entanglement to solve problems that classical computers cannot",
    "In the beginning there was nothing and then the universe expanded from a singularity",
    "The theory of relativity shows that space and time are curved by mass and energy",
    "Machine learning models can recognize patterns in data and make predictions about future events",
    "The history of Rome began as a small settlement on the Tiber River and grew into an empire",
    "Neural networks are composed of layers of interconnected nodes that process information",
    "The speed of light in vacuum is approximately three hundred million meters per second",
    "In mathematics a prime number is a natural number greater than one that cannot be formed by multiplying two smaller",
    "The human brain processes information through a network of billions of neurons connected by synapses",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose",
    "The Industrial Revolution transformed society by introducing machines that could produce goods faster than manual labor",
    "DNA contains the genetic instructions for the development and functioning of all known living organisms",
    "The Renaissance was a period of cultural artistic and scientific rebirth in Europe following the Middle Ages",
    "Climate change refers to long term shifts in temperature and weather patterns mainly caused by human activities",
]


@torch.no_grad()
def test_encoder(model, tokenizer, texts, use_window=False):
    lm_head = model.lm_head
    device = next(model.parameters()).device
    W_orig = lm_head.weight.detach().float().cpu().numpy().copy()
    out_f, in_f = W_orig.shape

    enc = WindowContextEncoder(d_model, context_dim, window_size=4) if use_window else None
    hl_cfg = HyperLoraConfig(context_dim=context_dim, rank=rank, in_features=in_f,
                              out_features=out_f, scale_min=0.3, scale_max=1.2, seed=0)

    total_base, total_mod = 0.0, 0.0
    deltas, scales, all_zs = [], [], []
    start_zs = []

    for text in texts:
        if enc: enc.reset()
        inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        ids = inp.input_ids.to(device)
        slen = ids.shape[1]
        if slen < 2: continue

        logits_base = model(ids).logits
        nll_pertok = torch.nn.functional.cross_entropy(
            logits_base[0, :-1].float(), ids[0, 1:], reduction="none").cpu().numpy()

        hs = model.transformer.wte(ids)
        hs = hs + model.transformer.wpe(torch.arange(slen, device=device))
        for block in model.transformer.h:
            hs = block(hs)[0]
        hs = model.transformer.ln_f(hs)

        for pos in range(slen - 1):
            h_pos = hs[0, pos].cpu().numpy().ravel().astype(np.float32)

            if use_window:
                ctx = enc.encode(h_pos)
            else:
                ctx = np.concatenate([
                    np.array([h_pos.mean(), h_pos.std(),
                              float(np.percentile(h_pos, 25)),
                              float(np.percentile(h_pos, 75)),
                              float(h_pos.max()), float(h_pos.min())], dtype=np.float32),
                    np.zeros(10, dtype=np.float32),
                ])[:context_dim]

            all_zs.append(ctx.copy())

            A, B, scale = hyperlora_generate(ctx, hl_cfg)
            W_np = lm_head.weight.detach().float().cpu().numpy()
            W_new, handle = apply_delta(W_np, A, B, scale)
            lm_head.weight.data = torch.from_numpy(W_new).to(device=device, dtype=lm_head.weight.dtype)
            logits_mod = model(ids[:, :pos+1]).logits
            nll_mod = torch.nn.functional.cross_entropy(
                logits_mod[0, -1].unsqueeze(0).float(), ids[0, pos+1].unsqueeze(0)).item()
            lm_head.weight.data = torch.from_numpy(revert_delta(W_new, handle)).to(
                device=device, dtype=lm_head.weight.dtype)

            nd = nll_mod - float(nll_pertok[pos])
            total_base += float(nll_pertok[pos])
            total_mod += nll_mod
            deltas.append(nd)
            scales.append(scale)

    drift = float(np.max(np.abs(lm_head.weight.detach().float().cpu().numpy() - W_orig)))
    z_sims = [float(np.dot(all_zs[i], all_zs[i-1]) / (
        np.linalg.norm(all_zs[i]) * np.linalg.norm(all_zs[i-1]) + 1e-8))
        for i in range(1, len(all_zs))]
    return {
        "encoder": "window(w=4)" if use_window else "heuristic",
        "n_tokens": len(deltas),
        "baseline_nll": total_base, "modified_nll": total_mod,
        "nll_delta": total_mod - total_base,
        "nll_delta_per_token": (total_mod - total_base) / max(1, len(deltas)),
        "fraction_improved": sum(1 for d in deltas if d < 0) / max(1, len(deltas)),
        "tokens_degraded_gt_0_1": sum(1 for d in deltas if d > 0.1),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)) if len(deltas) > 1 else 0.0,
        "max_delta": float(np.max(deltas)),
        "min_delta": float(np.min(deltas)),
        "mean_scale": float(np.mean(scales)),
        "z_sim_mean": float(np.mean(z_sims)) if z_sims else 0.0,
        "z_sim_std": float(np.std(z_sims)) if z_sims else 0.0,
        "reversibility_drift": drift,
    }


if __name__ == "__main__":
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    texts = LONG_TEXTS

    print(f"\n{'='*70}")
    print(f"  HEURISTIC vs WINDOW CONTEXT ENCODER")
    print(f"  {len(texts)} prompts, rank={rank}, context_dim={context_dim}")
    print(f"{'='*70}")

    for label, use_win in [("HEURISTIC (mean/std/percentile)", False),
                            ("WINDOW (avg 4 + random proj)", True)]:
        t0 = time.perf_counter()
        r = test_encoder(model, tokenizer, texts, use_window=use_win)
        dt = time.perf_counter() - t0

        print(f"\n  ── {label} ──")
        print(f"  Tokens:             {r['n_tokens']}")
        print(f"  NLL delta:          {r['nll_delta']:+8.4f}  ({r['nll_delta_per_token']:+8.6f}/tok)")
        print(f"  Fraction improved:  {r['fraction_improved']:.0%}")
        print(f"  Degraded >0.1:      {r['tokens_degraded_gt_0_1']}/{r['n_tokens']}")
        print(f"  Mean |Δ| per tok:   {r['mean_delta']:+.4f}")
        print(f"  Std(Δ):             {r['std_delta']:.4f}")
        print(f"  Δ range:            [{r['min_delta']:+.4f}, {r['max_delta']:+.4f}]")
        print(f"  Mean scale:         {r['mean_scale']:.3f}")
        print(f"  Z adjacent sim:     {r['z_sim_mean']:.4f} ± {r['z_sim_std']:.4f}")
        print(f"  Reversibility:      {r['reversibility_drift']:.2e}")
        print(f"  Time:               {dt:.1f}s")

        if label.startswith("HEURISTIC"):
            h = r
        else:
            w = r

    print(f"\n{'─'*70}")
    print(f"  COMPARISON")
    print(f"  Std(Δ):             {h['std_delta']:.4f} → {w['std_delta']:.4f}  ({w['std_delta']/max(1e-6,h['std_delta']):.1%}×)")
    print(f"  Degraded >0.1:      {h['tokens_degraded_gt_0_1']}/{h['n_tokens']} → {w['tokens_degraded_gt_0_1']}/{w['n_tokens']}")
    print(f"  NLL Δ total:        {h['nll_delta']:+.4f} → {w['nll_delta']:+.4f}")
    print(f"  Z stability:        {h['z_sim_mean']:.4f} → {w['z_sim_mean']:.4f}")
    print(f"{'─'*70}")

    os.makedirs("quality/context_encoder", exist_ok=True)
    with open("quality/context_encoder/heuristic_vs_window.json", "w") as f:
        json.dump({"heuristic": h, "window": w}, f, indent=2, default=str)
