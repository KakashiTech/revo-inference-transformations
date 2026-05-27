from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules

@torch.no_grad()
def spectral_prune_tensor(W: torch.Tensor, energy_keep: float = 0.9) -> Tuple[torch.Tensor, int, int]:
    device = W.device
    dtype = W.dtype
    X = W.detach().to(device="cpu", dtype=torch.float32)
    F = torch.fft.rfft2(X)
    P = (F.real * F.real + F.imag * F.imag).reshape(-1)
    total = float(torch.sum(P).item())
    if total <= 0.0 or P.numel() == 0:
        Y = torch.fft.irfft2(F, s=X.shape)
        return Y.to(device=device, dtype=dtype), int(P.numel()), int(P.numel())
    vals, idx = torch.sort(P, descending=True)
    cumsum = torch.cumsum(vals, dim=0)
    target = float(max(0.0, min(1.0, energy_keep))) * total
    k = int(torch.searchsorted(cumsum, torch.tensor(target)).item()) + 1
    k = int(max(1, min(k, P.numel())))
    mask = torch.zeros_like(P, dtype=torch.bool)
    mask[:k] = True
    mask_full = torch.zeros_like(F, dtype=torch.bool).reshape(-1)
    mask_full[:] = mask
    mask_full = mask_full.reshape(F.shape)
    F_pruned = torch.where(mask_full, F, torch.zeros_like(F))
    Y = torch.fft.irfft2(F_pruned, s=X.shape)
    return Y.to(device=device, dtype=dtype), int(k), int(P.numel())


@torch.no_grad()
def prune_model_spectral(
    model: nn.Module,
    energy_keep: float = 0.9,
    name_patterns: List[str] | None = None,
    skip_lm_head: bool = True,
) -> Dict[str, Dict[str, float]]:
    pats = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    report: Dict[str, Dict[str, float]] = {}

    head_module = None
    input_embed_weight = None
    try:
        get_head = getattr(model, "get_output_embeddings", None)
        if callable(get_head):
            head_module = get_head()
    except Exception:
        head_module = None
    try:
        get_inp = getattr(model, "get_input_embeddings", None)
        if callable(get_inp):
            inp = get_inp()
            if inp is not None and hasattr(inp, "weight"):
                input_embed_weight = getattr(inp, "weight")
    except Exception:
        input_embed_weight = None

    for name, m in model.named_modules():
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in pats):
            continue
        if not hasattr(m, "weight") or not isinstance(getattr(m, "weight"), torch.Tensor):
            continue
        W = getattr(m, "weight")
        if W.dim() != 2:
            continue
        try:
            if (input_embed_weight is not None) and (W is input_embed_weight):
                continue
        except Exception:
            pass
        Y, kept, total = spectral_prune_tensor(W, energy_keep=energy_keep)
        orig_norm = float(W.detach().norm().item())
        setattr(m, "weight", nn.Parameter(Y.to(device=W.device, dtype=W.dtype), requires_grad=False))
        report[name] = {
            "kept_coeffs": float(kept),
            "total_coeffs": float(total),
            "kept_ratio": float(kept) / float(total) if total > 0 else 0.0,
            "orig_norm": orig_norm,
            "pruned_norm": float(Y.detach().norm().item()),
        }
    return report
