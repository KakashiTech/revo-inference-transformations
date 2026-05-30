"""FieldSelector (Φ/A/C) — benchmark de activación selectiva por token.

Demuestra:
  - El field selector descubre módulos del modelo (capas + lm_head)
  - Por cada token, selecciona qué módulos activar y con qué intensidad
  - La distribución de activaciones varía según el contexto (Z)
  - Tokens simples vs complejos activan distinto número de módulos
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revo._logging import get_logger
from revo.field_selector import FieldSelector, FieldSelectorConfig

log = get_logger(__name__)

TEXTS = [
    "The capital of France is Paris",
    "Quantum computing relies on superposition",
    "In the beginning there was nothing",
    "The theory of relativity shows that space and time are curved",
    "cat sat on the mat",
    "Supercalifragilisticexpialidocious is a very long word",
    "Neural networks are composed of layers of interconnected nodes",
    "The human brain processes information through a network of billions of neurons",
    "DNA contains the genetic instructions for the development and functioning of all known living organisms",
    "Climate change refers to long term shifts in temperature and weather patterns",
]

SIMPLE = [
    "cat sat",
    "dog runs",
    "it is",
    "yes no",
    "go stop",
    "up down",
    "red blue",
    "big small",
]

COMPLEX = [
    "The fundamental axioms of non-Euclidean geometry challenge the intuitive notion of parallel lines",
    "The mitochondrial genome encodes thirteen proteins essential for oxidative phosphorylation",
    "The covariance matrix of a multivariate Gaussian distribution must be positive semidefinite",
    "Entanglement entropy in conformal field theories follows the Ryu-Takayanagi formula",
    "The categorical imperative requires that one act only according to maxims universalizable without contradiction",
    "Superconductivity at room temperature would revolutionize energy transmission and storage",
    "The Riemann hypothesis concerns the distribution of nontrivial zeros of the zeta function",
    "Neuroplasticity enables the reorganization of synaptic connections in response to learning and injury",
]


@torch.no_grad()
def compute_Z_for_token(model, tokenizer, text: str, pos: int):
    device = next(model.parameters()).device
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    ids = inp.input_ids.to(device)
    slen = ids.shape[1]
    if pos >= slen - 1:
        pos = slen - 2

    hs = model.transformer.wte(ids)
    hs = hs + model.transformer.wpe(torch.arange(slen, device=device))
    for block in model.transformer.h:
        hs = block(hs)[0]
    hs = model.transformer.ln_f(hs)

    h = hs[0, pos].detach().float().cpu().numpy().ravel()

    from revo.context_encoder import WindowContextEncoder
    enc = WindowContextEncoder(d_model=h.shape[0], context_dim=16, window_size=4, novelty=True)
    Z = enc.encode(h)
    return Z, ids[0, pos].item(), ids[0, pos + 1].item()


@torch.no_grad()
def analyze_text(fs: FieldSelector, model, tokenizer, text: str, label: str = ""):
    device = next(model.parameters()).device
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    ids = inp.input_ids.to(device)
    slen = ids.shape[1]
    if slen < 2:
        return None

    token_reports = []
    for pos in range(slen - 1):
        Z, tok_id, next_id = compute_Z_for_token(model, tokenizer, text, pos)
        selected = fs.select(Z)
        dist = fs.get_active_distribution(Z)

        token_str = tokenizer.decode([tok_id])
        next_str = tokenizer.decode([next_id])
        token_reports.append({
            "pos": pos,
            "token": token_str,
            "next": next_str,
            "n_selected": dist["n_selected"],
            "mean_intensity": dist["mean_intensity"],
            "layer_dist": dist["layer_distribution"],
            "modules": dist["selected_modules"],
        })

    avg_selected = np.mean([t["n_selected"] for t in token_reports])
    avg_intensity = np.mean([t["mean_intensity"] for t in token_reports])

    # Aggregate layer distribution
    layer_counts: dict = {}
    for t in token_reports:
        for k, v in t["layer_dist"].items():
            layer_counts[k] = layer_counts.get(k, 0) + v

    summary = {
        "text": text[:60],
        "label": label,
        "n_tokens": len(token_reports),
        "avg_selected_modules": float(avg_selected),
        "avg_intensity": float(avg_intensity),
        "layer_activation_counts": layer_counts,
        "token_details": token_reports,
    }
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FieldSelector Analysis")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-active", type=int, default=0)
    ap.add_argument("--results-json", default="quality/field_selector/analysis.json")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    cfg = FieldSelectorConfig(
        context_dim=16,
        embed_dim=8,
        hidden_dim=64,
        max_active_modules=args.max_active,
    )
    fs = FieldSelector(model, cfg)

    print(f"Model: {args.model}")
    print(f"Discovered {fs.n_modules} modules:")
    for name in fs.stats()["module_names"]:
        print(f"  {name}")
    print()

    # Analyze regular texts
    all_reports = []
    print("=" * 65)
    print("  ANALYZING TEXTS")
    print("=" * 65)
    for text in TEXTS:
        report = analyze_text(fs, model, tokenizer, text, "regular")
        if report:
            all_reports.append(report)
            print(f"\n  Text: {text[:50]}...")
            print(f"    Tokens: {report['n_tokens']}")
            print(f"    Avg selected: {report['avg_selected_modules']:.1f} modules")
            print(f"    Avg intensity: {report['avg_intensity']:.3f}")
            sorted_layers = sorted(report["layer_activation_counts"].items(), key=lambda x: -x[1])
            top5 = sorted_layers[:5]
            print(f"    Top layers: {', '.join(f'{k}({v})' for k, v in top5)}")

    # Compare simple vs complex
    print("\n" + "=" * 65)
    print("  COMPARING SIMPLE VS COMPLEX TOKENS")
    print("=" * 65)

    simple_reports = []
    for text in SIMPLE:
        r = analyze_text(fs, model, tokenizer, text, "simple")
        if r:
            simple_reports.append(r)

    complex_reports = []
    for text in COMPLEX:
        r = analyze_text(fs, model, tokenizer, text, "complex")
        if r:
            complex_reports.append(r)

    avg_simple = np.mean([r["avg_selected_modules"] for r in simple_reports]) if simple_reports else 0
    avg_complex = np.mean([r["avg_selected_modules"] for r in complex_reports]) if complex_reports else 0
    avg_simple_int = np.mean([r["avg_intensity"] for r in simple_reports]) if simple_reports else 0
    avg_complex_int = np.mean([r["avg_intensity"] for r in complex_reports]) if complex_reports else 0

    print(f"\n  Simple tokens ({len(simple_reports)} texts):")
    print(f"    Avg selected modules: {avg_simple:.2f}")
    print(f"    Avg intensity:        {avg_simple_int:.4f}")
    print(f"\n  Complex tokens ({len(complex_reports)} texts):")
    print(f"    Avg selected modules: {avg_complex:.2f}")
    print(f"    Avg intensity:        {avg_complex_int:.4f}")
    print(f"\n  Ratio (complex/simple): {avg_complex / max(avg_simple, 0.01):.2f}x modules, "
          f"{avg_complex_int / max(avg_simple_int, 0.001):.2f}x intensity")

    # Save
    combined = {
        "config": asdict(cfg),
        "model": args.model,
        "n_modules": fs.n_modules,
        "module_names": fs.stats()["module_names"],
        "texts": all_reports,
        "simple_vs_complex": {
            "simple_avg_modules": avg_simple,
            "complex_avg_modules": avg_complex,
            "simple_avg_intensity": avg_simple_int,
            "complex_avg_intensity": avg_complex_int,
            "ratio_modules_simple": avg_complex / max(avg_simple, 0.01),
            "ratio_intensity_simple": avg_complex_int / max(avg_simple_int, 0.001),
        },
    }
    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info("Saved to %s", args.results_json)
