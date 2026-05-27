from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules

def _orient_weight_bias(module: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, bool]:
    """Return (W_oriented, b, in_features, out_features, transposed)
    with W_oriented in shape [out, in].
    """
    assert hasattr(module, "weight"), "module must have .weight"
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    if not isinstance(W, torch.Tensor) or W.dim() != 2:
        raise AssertionError("module.weight must be 2D tensor")
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    W_det = W.detach()
    if b_det is not None:
        if W_det.shape[0] == b_det.numel():
            return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False
        if W_det.shape[1] == b_det.numel():
            return W_det.t(), b_det, int(W_det.shape[0]), int(W_det.shape[1]), True
    # Fallback assume already [out,in]
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False


@torch.no_grad()
def morse_skeletonize(
    model: nn.Module,
    keep_frac: float = 0.95,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, Dict[str, int]]:
    """Apply a discrete-Morse inspired skeletonization by zeroing least-influential
    output units per 2D-weight module.

    - Compute per-output score as L2 norm of oriented weight rows.
    - Keep top ceil(keep_frac * out_features) units, zero the rest (weights and bias rows/entries).
    - Returns a report mapping module name -> {"out": out_features, "kept": kept, "pruned": pruned}.
    """
    patterns = name_patterns or ["mlp", "c_fc", "c_proj", "attn"]
    keep_frac = float(max(0.0, min(1.0, keep_frac)))
    report: Dict[str, Dict[str, int]] = {}

    for name, m in model.named_modules():
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in patterns):
            continue
        W = getattr(m, "weight", None)
        if not isinstance(W, torch.Tensor) or W.dim() != 2:
            continue
        if "embedding" in m.__class__.__name__.lower():
            continue
        W_o, b_o, in_f, out_f, transposed = _orient_weight_bias(m)
        # scores on rows
        scores = torch.norm(W_o, dim=1)
        kept = int((scores.numel() * keep_frac) + 0.9999)
        kept = int(max(1, min(scores.numel(), kept)))
        # indices to keep
        topk = torch.topk(scores, kept, largest=True).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[topk] = True
        # apply mask: zero pruned rows in oriented space
        W_new = W_o.clone()
        W_new[~mask, :] = 0.0
        b_new = b_o.clone() if isinstance(b_o, torch.Tensor) else None
        if isinstance(b_new, torch.Tensor):
            b_new[~mask] = 0.0
        # write back with orientation
        if transposed:
            m.weight.data.copy_(W_new.t().to(dtype=m.weight.dtype, device=m.weight.device))
        else:
            m.weight.data.copy_(W_new.to(dtype=m.weight.dtype, device=m.weight.device))
        if isinstance(b_new, torch.Tensor) and hasattr(m, "bias") and isinstance(m.bias, torch.Tensor):
            m.bias.data.copy_(b_new.to(dtype=m.bias.dtype, device=m.bias.device))
        pruned = int(out_f - kept)
        report[name] = {"out": int(out_f), "kept": int(kept), "pruned": int(pruned)}
    return report
