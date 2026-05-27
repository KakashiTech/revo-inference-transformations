#!/usr/bin/env python3
"""Experimento espectral: REVO preserva estructura de activaciones vs quant+prune la destruye.

Mide la distribucion espectral (SVD) de las activaciones hidden antes/despues
de cada transformacion. La hipotesis es que REVO preserva el espectro (los
autovalores decaen igual) mientras que quant+prune lo distorsiona.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import encode_text, evaluate_nll, seed_everything

from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank

try:
    import torch.nn.utils.prune as prune
except Exception:
    prune = None


MODEL = "sshleifer/tiny-gpt2"
TEXTS = [
    "La transformacion espectral revela la estructura interna de las representaciones.",
    "Los autovalores del embedding definen la metrica del espacio latente.",
    "La poda por magnitud destruye componentes espectrales de baja energia.",
    "REVO modula sin alterar la topologia del espacio de activaciones.",
    "La reversibilidad implica que la informacion no se pierde, solo se transforma.",
]
SEED = 42
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_module(model, path: str) -> nn.Module:
    """Resolve a dotted module path like 'transformer.h.0' or 'transformer.ln_f'."""
    cur = model
    for part in path.split(".") if path else []:
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _collect_activations(model, tok, texts: List[str], max_len: int, module_path: str) -> torch.Tensor:
    """Collect hidden states from a specific module (e.g. 'transformer.h.0' or 'transformer.ln_f')."""
    model.eval()
    activations: List[torch.Tensor] = []

    def hook_fn(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        activations.append(h.detach().float().cpu())

    mod = _resolve_module(model, module_path)
    handle = mod.register_forward_hook(hook_fn)

    with torch.no_grad():
        for t in texts:
            batch = encode_text(tok, t, max_len, device=DEVICE)
            model(**batch)

    handle.remove()
    return torch.cat(activations, dim=1).squeeze(0)  # [T, H] over all tokens


def _svd_spectrum(acts: torch.Tensor) -> np.ndarray:
    """Return singular values (normalized) of the activation matrix."""
    U, S, Vh = torch.linalg.svd(acts.float(), full_matrices=False)
    S_np = S.cpu().numpy()
    return S_np / S_np.sum()  # normalized spectrum


def _spectral_entropy(S: np.ndarray) -> float:
    """Shannon entropy of the normalized spectrum."""
    S = S[S > 0]
    return float(-np.sum(S * np.log(S)))


def _spectral_distance(S1: np.ndarray, S2: np.ndarray) -> float:
    """L2 distance between normalized spectra."""
    min_len = min(len(S1), len(S2))
    return float(np.linalg.norm(S1[:min_len] - S2[:min_len]))


def experiment() -> Dict[str, Any]:
    seed_everything(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results: Dict[str, Any] = {"model": MODEL, "variants": {}}

    for module_path in ["transformer.ln_f", "transformer.h.0", "transformer.h.1"]:
        print(f"\n--- {module_path} ---")

        # --- Baseline activations ---
        model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
        model.eval()
        base_acts = _collect_activations(model, tok, TEXTS, MAX_LEN, module_path)
        base_spectrum = _svd_spectrum(base_acts)
        base_entropy = _spectral_entropy(base_spectrum)
        base_nll = evaluate_nll(model, tok, TEXTS, max_length=MAX_LEN, device=DEVICE)
        print(f"  baseline:  entropy={base_entropy:.4f}  nll={base_nll:.6f}")

        # --- REVO activations ---
        model_revo = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
        model_revo.eval()
        prof = profile_model_2d(model_revo, name_patterns=["attn", "mlp", "c_fc", "c_proj"], max_rank=None)
        ranks = allocate_ranks_energy_with_caps(prof, energy_keep=0.92, max_rank=None, max_rank_frac=0.20)
        replace_2d_modules_with_lowrank(model_revo, ranks, calibrate=True, calibrate_samples=128, seed=SEED)
        revo_acts = _collect_activations(model_revo, tok, TEXTS, MAX_LEN, module_path)
        revo_spectrum = _svd_spectrum(revo_acts)
        revo_entropy = _spectral_entropy(revo_spectrum)
        revo_nll = evaluate_nll(model_revo, tok, TEXTS, max_length=MAX_LEN, device=DEVICE)
        print(f"  revo:      entropy={revo_entropy:.4f}  nll={revo_nll:.6f}  dist={_spectral_distance(base_spectrum, revo_spectrum):.6f}")

        # --- Quant+Prune activations ---
        model_qp = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
        model_qp.eval()
        if prune is not None:
            params = [(mod, "weight") for _, mod in model_qp.named_modules()
                      if hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), nn.Parameter)]
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.3)
            for mod, _ in params:
                try:
                    prune.remove(mod, "weight")
                except Exception:
                    pass
        qp_acts = _collect_activations(model_qp, tok, TEXTS, MAX_LEN, module_path)
        qp_spectrum = _svd_spectrum(qp_acts)
        qp_entropy = _spectral_entropy(qp_spectrum)
        qp_nll = evaluate_nll(model_qp, tok, TEXTS, max_length=MAX_LEN, device=DEVICE)
        print(f"  q+prune:   entropy={qp_entropy:.4f}  nll={qp_nll:.6f}  dist={_spectral_distance(base_spectrum, qp_spectrum):.6f}")

        label = module_path.replace(".", "_")
        results["variants"][label] = {
            "baseline": {"nll": base_nll, "spectral_entropy": base_entropy, "spectrum": base_spectrum.tolist()[:20]},
            "revo": {"nll": revo_nll, "spectral_entropy": revo_entropy, "spectral_distance": _spectral_distance(base_spectrum, revo_spectrum),
                      "spectrum": revo_spectrum.tolist()[:20]},
            "quant_prune": {"nll": qp_nll, "spectral_entropy": qp_entropy, "spectral_distance": _spectral_distance(base_spectrum, qp_spectrum),
                             "spectrum": qp_spectrum.tolist()[:20]},
        }

    return results


def main():
    print("=" * 70)
    print("EXPERIMENTO ESPECTRAL: REVO preserva estructura vs quant+prune la destruye")
    print("=" * 70)

    results = experiment()

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for layer_key, variants in results["variants"].items():
        print(f"\n  {layer_key}:")
        b = variants["baseline"]
        r = variants["revo"]
        q = variants["quant_prune"]
        print(f"    baseline entropy:           {b['spectral_entropy']:.6f}")
        print(f"    revo entropy:               {r['spectral_entropy']:.6f}  (delta={r['spectral_entropy'] - b['spectral_entropy']:+.6f})")
        print(f"    quant+prune entropy:         {q['spectral_entropy']:.6f}  (delta={q['spectral_entropy'] - b['spectral_entropy']:+.6f})")
        print(f"    revo spectral distance:      {r['spectral_distance']:.6f}")
        print(f"    quant+prune spectral dist:   {q['spectral_distance']:.6f}")

    os.makedirs("quality/experiments", exist_ok=True)
    with open("quality/experiments/spectral_preservation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to quality/experiments/spectral_preservation.json")


if __name__ == "__main__":
    main()
