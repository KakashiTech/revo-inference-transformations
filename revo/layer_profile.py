from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def get_linear_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Linear]]:
    """Return only nn.Linear modules (exclude Embedding/others)."""
    layers: List[Tuple[str, torch.nn.Linear]] = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and getattr(m, "weight", None) is not None and m.weight.dim() == 2:
            layers.append((name, m))
    return layers


def spectral_profile_linear(module: torch.nn.Linear, max_rank: int | None = None) -> Dict[str, np.ndarray | float | int]:
    W = module.weight.detach().float().cpu()
    s = torch.linalg.svdvals(W)
    if max_rank is not None:
        s = s[: int(max_rank)]
    s2 = (s * s).cpu().numpy()
    total = float(np.sum(s2)) if s2.size > 0 else 0.0
    if total <= 0:
        energy = np.zeros_like(s2)
        eff_rank = 0.0
        stable_rank = 0.0
        h_norm = 0.0
    else:
        energy = np.cumsum(s2) / total
        p = s2 / total
        p = p[p > 0]
        h = float(-np.sum(p * np.log(p + 1e-12))) if p.size > 0 else 0.0
        eff_rank = float(np.exp(h))
        stable_rank = float(total / float(np.max(s2))) if s2.size > 0 else 0.0
        h_norm = float(h / np.log(float(len(s2)) + 1e-12)) if len(s2) > 1 else 0.0
    return {
        "out_features": int(module.weight.shape[0]),
        "in_features": int(module.weight.shape[1]),
        "num_singular": int(len(s2)),
        "singular_values": s.cpu().numpy(),
        "energy_cumsum": energy,
        "energy_total": total,
        "effective_rank": eff_rank,
        "stable_rank": stable_rank,
        "entropy_norm": h_norm,
    }


def profile_model(model: torch.nn.Module, max_rank: int | None = None) -> Dict[str, Dict[str, np.ndarray | float | int]]:
    prof: Dict[str, Dict[str, np.ndarray | float | int]] = {}
    for name, m in get_linear_layers(model):
        prof[name] = spectral_profile_linear(m, max_rank=max_rank)
    return prof


def allocate_ranks_by_energy(profile: Dict[str, Dict[str, np.ndarray | float | int]], energy_keep: float = 0.98, max_rank: int | None = None) -> Dict[str, int]:
    alloc: Dict[str, int] = {}
    for name, p in profile.items():
        ec = p["energy_cumsum"]  # type: ignore[index]
        if not isinstance(ec, np.ndarray) or ec.size == 0:
            alloc[name] = 0
            continue
        r = int(np.searchsorted(ec, min(0.9999, float(energy_keep))) + 1)
        if max_rank is not None:
            r = min(r, int(max_rank))
        # always cap to min(in,out)
        of, inf = int(p["out_features"]) , int(p["in_features"])  # type: ignore[index]
        r = int(max(1, min(r, of, inf)))
        alloc[name] = r
    return alloc


def _max_beneficial_rank(out_features: int, in_features: int) -> int:
    """Maximum rank that does not increase parameter count compared to dense.

    Dense params: of*inf (+bias)
    Low-rank params: of*r + r*inf (+bias)
    Require: of*r + r*inf <= of*inf  =>  r <= (of*inf)/(of+inf)
    """
    of = int(max(1, out_features))
    inf = int(max(1, in_features))
    return int(max(0, (of * inf) // (of + inf)))


def allocate_ranks_energy_with_caps(
    profile: Dict[str, Dict[str, np.ndarray | float | int]],
    energy_keep: float = 0.92,
    max_rank: int | None = None,
    max_rank_frac: float = 0.25,
) -> Dict[str, int]:
    """Allocate per-layer ranks by spectral energy while enforcing parameter-saving caps.

    - energy_keep: target fraction of spectral energy to retain.
    - max_rank: hard global cap on rank (optional).
    - max_rank_frac: per-layer cap as fraction of min(out,in) to prevent oversized ranks.

    Only assigns a rank if it yields parameter savings; otherwise sets rank=0 (skip).
    """
    alloc: Dict[str, int] = {}
    for name, p in profile.items():
        try:
            of = int(p["out_features"])  # type: ignore[index]
            inf = int(p["in_features"])  # type: ignore[index]
            ec = p["energy_cumsum"]  # type: ignore[index]
        except Exception:
            alloc[name] = 0
            continue
        if not isinstance(ec, np.ndarray) or ec.size == 0 or of <= 0 or inf <= 0:
            alloc[name] = 0
            continue
        # Energy-based rank
        r_e = int(np.searchsorted(ec, min(0.9999, float(energy_keep))) + 1)
        # Per-layer caps
        r_cap_frac = int(max(1, np.floor(float(min(of, inf)) * float(max_rank_frac))))
        r_cap_benefit = _max_beneficial_rank(of, inf)
        r_cap_global = int(max_rank) if max_rank is not None else r_e
        r_cap = max(1, min(r_cap_frac, r_cap_benefit, r_cap_global))
        r = int(max(1, min(r_e, r_cap)))
        # Ensure actual savings; if not, skip (rank=0)
        if (of * r + r * inf) >= (of * inf):
            r = 0
        alloc[name] = r
    return alloc


# ---- Extensions for generic 2D-weight modules (e.g., GPT-2 Conv1D) ----
def get_2d_weight_modules(model: torch.nn.Module, name_patterns: list[str] | None = None) -> list[tuple[str, torch.nn.Module]]:
    """Return modules with 2D weights filtered by name patterns, excluding Embedding.

    Includes GPT-2 Conv1D-like layers (weight is 2D) and skips Embedding and modules without 2D tensor weights.
    """
    pats = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    layers: list[tuple[str, torch.nn.Module]] = []
    for name, m in model.named_modules():
        if not any(p in name for p in pats):
            continue
        W = getattr(m, "weight", None)
        if isinstance(W, torch.Tensor) and W.dim() == 2:
            # Skip common embedding modules
            if "embedding" in m.__class__.__name__.lower():
                continue
            layers.append((name, m))
    return layers


def spectral_profile_module_2d(module: torch.nn.Module, max_rank: int | None = None) -> Dict[str, np.ndarray | float | int]:
    W = getattr(module, "weight")
    Wf = W.detach().float().cpu()
    s = torch.linalg.svdvals(Wf)
    if max_rank is not None:
        s = s[: int(max_rank)]
    s2 = (s * s).cpu().numpy()
    total = float(np.sum(s2)) if s2.size > 0 else 0.0
    if total <= 0:
        energy = np.zeros_like(s2)
        eff_rank = 0.0
        stable_rank = 0.0
        h_norm = 0.0
    else:
        energy = np.cumsum(s2) / total
        p = s2 / total
        p = p[p > 0]
        h = float(-np.sum(p * np.log(p + 1e-12))) if p.size > 0 else 0.0
        eff_rank = float(np.exp(h))
        stable_rank = float(total / float(np.max(s2))) if s2.size > 0 else 0.0
        h_norm = float(h / np.log(float(len(s2)) + 1e-12)) if len(s2) > 1 else 0.0
    return {
        "out_features": int(Wf.shape[0]),
        "in_features": int(Wf.shape[1]),
        "num_singular": int(len(s2)),
        "singular_values": s.cpu().numpy(),
        "energy_cumsum": energy,
        "energy_total": total,
        "effective_rank": eff_rank,
        "stable_rank": stable_rank,
        "entropy_norm": h_norm,
    }


def profile_model_2d(model: torch.nn.Module, name_patterns: list[str] | None = None, max_rank: int | None = None) -> Dict[str, Dict[str, np.ndarray | float | int]]:
    prof: Dict[str, Dict[str, np.ndarray | float | int]] = {}
    for name, m in get_2d_weight_modules(model, name_patterns=name_patterns):
        prof[name] = spectral_profile_module_2d(m, max_rank=max_rank)
    return prof
