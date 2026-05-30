#!/usr/bin/env python3
"""Primitiva Router — experimento con GPT-2 real.

Mide:
  - Perplejidad antes/después de entrenar el router
  - Distribución de primitivas por capa
  - Variación por token
  - Correlación con categorías lingüísticas
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo.primitiva_router import PrimitiveModel

DEVICE = torch.device("cpu")
MODEL_NAME = "sshleifer/tiny-gpt2"
TEXTS = [
    "The capital of France is Paris and it has been for centuries.",
    "Quantum computing relies on superposition and entanglement.",
    "The theory of relativity shows that space and time are curved.",
    "Neural networks are composed of layers of interconnected nodes.",
    "The human brain processes information through billions of neurons.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "DNA contains the genetic instructions for all living organisms.",
    "The Renaissance was a period of cultural and scientific rebirth.",
    "Climate change refers to long term shifts in temperature patterns.",
    "Machine learning models can recognize patterns in data.",
    "The speed of light in vacuum is approximately 300 million m/s.",
    "In mathematics a prime number is greater than one.",
]


def compute_ppl(model, tok, texts, max_len=64):
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
            out = model(**enc)
            logits = out.logits[0]  # [T, V]
            shift_logits = logits[:-1]
            shift_labels = enc.input_ids[0, 1:]
            loss = F.cross_entropy(shift_logits, shift_labels)
            total_nll += loss.item() * (shift_labels.numel())
            total_tokens += shift_labels.numel()
    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll), avg_nll


def main():
    print("=" * 65)
    print("REVO Primitiva Router — GPT-2 experiment")
    print("=" * 65)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    # --- Baseline PPL ---
    print("\n[1] Baseline (modelo original)")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    base_ppl, base_nll = compute_ppl(base, tok, TEXTS)
    print(f"  PPL: {base_ppl:.2f} | NLL: {base_nll:.4f}")
    n_base = sum(p.numel() for p in base.parameters())
    print(f"  Params: {n_base:,}")

    # --- Wrap with PrimitiveModel (soft routing for training) ---
    print("\n[2] Inicializando PrimitiveModel...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    pm = PrimitiveModel(
        model,
        enabled=["dense", "circulant", "wdm", "holography", "lowrank"],
        name_patterns=["c_attn", "c_proj", "c_fc"],
        router_hard=True,  # Gumbel-Softmax during training
    )
    d = pm.describe()
    print(f"  Capas reemplazadas: {d['replaced_layers']}")
    n_router = sum(p.numel() for n, p in pm.named_parameters() if 'router' in n)
    n_total = sum(p.numel() for p in pm.parameters())
    print(f"  Router params: {n_router:,} | Total params: {n_total:,}")
    print(f"  Overhead: {n_router / n_base * 100:.2f}%")

    # Peek at which layers got which primitives
    for n, sel in pm._selectors.items():
        p = sel.describe()["primitives"]
        shape_info = f"{sel.in_features}→{sel.out_features}"
        print(f"    {n:<40s} {shape_info:>8s}  {p}")

    # --- PPL before training (random router) ---
    pm.eval()
    init_ppl, init_nll = compute_ppl(pm, tok, TEXTS)
    print(f"\n[3] PPL inicial (router aleatorio): {init_ppl:.2f} (Δ={init_ppl-base_ppl:+.2f})")

    # --- Train router ---
    print("\n[4] Entrenando router (200 steps)...")
    pm.train()
    opt = torch.optim.AdamW(
        [p for n, p in pm.named_parameters() if 'router' in n],
        lr=3e-3,
    )
    all_text = TEXTS * 5
    for step in range(201):
        opt.zero_grad()
        for t in all_text:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=64)
            out = pm(**enc)
            logits = out.logits[0]
            shift_logits = logits[:-1]
            shift_labels = enc.input_ids[0, 1:]
            loss = F.cross_entropy(shift_logits, shift_labels)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(pm.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            pm.eval()
            ppl, nll = compute_ppl(pm, tok, TEXTS)
            pm.train()
            print(f"  step {step:3d} | PPL={ppl:.2f} | NLL={nll:.4f} | Δ={ppl-base_ppl:+.2f}")

    # --- Final PPL ---
    pm.eval()
    final_ppl, final_nll = compute_ppl(pm, tok, TEXTS)
    print(f"\n[5] PPL final: {final_ppl:.2f} (Δ={final_ppl-base_ppl:+.2f})")
    print(f"    Inicial: {init_ppl:.2f} → Final: {final_ppl:.2f}")

    # --- Router analysis ---
    print("\n[6] Análisis del router por capa")
    with torch.no_grad():
        all_tokens = torch.cat([
            tok(t, return_tensors="pt", truncation=True, max_length=64).input_ids
            for t in TEXTS
        ], dim=1)  # [1, total_seq]
        _ = pm(all_tokens)
        weights = pm.collect_router_weights()

    prim_names = ["dense", "circ", "wdm", "holo", "lora"]
    header = f"{'Capa':<40s}"
    for pn in prim_names:
        header += f" {pn:>6s}"
    print(header)
    print("-" * len(header))
    for name, w in weights.items():
        probs = w.mean(dim=tuple(range(w.ndim - 1)))
        row = f"{name:<40s}"
        for i in range(len(probs)):
            row += f" {probs[i].item()*100:>5.1f}%"
        print(row)

    # --- Token-level variation ---
    print("\n[7] Variación por token (primer prompt)")
    prompt = TEXTS[0]
    enc = tok(prompt, return_tensors="pt")
    _ = pm(enc.input_ids)
    w = pm.collect_router_weights()
    first_layer = list(w.keys())[0]
    w_first = w[first_layer][0]  # [T, N]
    tokens = enc.input_ids[0]
    n_show = min(10, len(tokens))
    print(f"  Capa: {first_layer}")
    for ti in range(n_show):
        sel = w_first[ti].argmax().item()
        sel_name = prim_names[sel] if sel < len(prim_names) else "?"
        token_str = tok.decode(tokens[ti].item())
        parts = [f"{prim_names[i]}={w_first[ti,i]:.2f}"
                 for i in range(w_first.shape[-1])]
        print(f"    tok {ti:2d} '{token_str:>8s}' → {sel_name:>10s}  ({', '.join(parts)})")

    print("\n" + "=" * 65)
    print(f"Resumen: router {n_router:,} params | "
          f"PPL {init_ppl:.2f}→{final_ppl:.2f} (baseline {base_ppl:.2f})")
    print("=" * 65)


if __name__ == "__main__":
    main()
