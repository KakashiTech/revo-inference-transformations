from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from revo._utils import set_by_name, skip_tied_weights

def _orient_weight_bias(module: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, bool]:
    """Return oriented weight W_o in shape [out, in] and bias b if present.

    Also returns (in_features, out_features, transposed_flag) where transposed indicates
    if the original storage had to be transposed to get [out, in].
    """
    assert hasattr(module, "weight"), "module must have .weight"
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    if not isinstance(W, torch.Tensor) or W.dim() != 2:
        raise AssertionError("module.weight must be 2D tensor")
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    W_det = W.detach()
    # If bias matches first dim -> already [out, in]
    if b_det is not None:
        if W_det.shape[0] == b_det.numel():
            return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False
        if W_det.shape[1] == b_det.numel():
            return W_det.t(), b_det, int(W_det.shape[0]), int(W_det.shape[1]), True
    # Fallback assume [out, in]
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False


class SpectralLinearLike(nn.Module):
    def __init__(self, base: nn.Module, rank: int = 4):
        super().__init__()
        W_o, b_o, in_f, out_f, _ = _orient_weight_bias(base)
        r = int(max(1, min(rank, min(out_f, in_f))))
        U, S, Vh = torch.linalg.svd(W_o.to(device="cpu", dtype=torch.float32), full_matrices=False)
        U_r = U[:, :r].contiguous()
        S_r = S[:r].contiguous()
        V_r = Vh[:r, :].t().contiguous()
        dev = getattr(base.weight, "device", torch.device("cpu"))
        dt = getattr(base.weight, "dtype", torch.float32)
        self.U = nn.Parameter(U_r.to(device=dev, dtype=dt), requires_grad=False)   # [out, r]
        self.S = nn.Parameter(S_r.to(device=dev, dtype=dt), requires_grad=False)   # [r]
        self.V = nn.Parameter(V_r.to(device=dev, dtype=dt), requires_grad=False)   # [in, r]
        self.out_features = out_f
        self.in_features = in_f
        if isinstance(b_o, torch.Tensor):
            self.bias = nn.Parameter(b_o.to(device=dev, dtype=dt), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collapse to 2D [N, in], operate on last dim, then restore
        orig_shape = x.shape
        in_f = self.in_features
        x2 = x.view(-1, orig_shape[-1])
        # x2: [N, in] -> [N, r]
        h = x2 @ self.V
        h = h * self.S
        y2 = h @ self.U.t()  # [N, out]
        if self.bias is not None:
            y2 = y2 + self.bias
        y = y2.view(*orig_shape[:-1], self.out_features)
        return y


def _set_by_name(root: nn.Module, path: str, new_mod: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else getattr(parent, "_modules")[p]
    last = parts[-1]
    if last.isdigit():
        parent._modules[last] = new_mod
    else:
        setattr(parent, last, new_mod)


def replace_mlp_with_spectral(
    model: nn.Module,
    rank: int = 4,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, float]:
    pats = name_patterns or ["mlp", "c_fc", "c_proj"]
    modules = 0
    orig_params = 0
    coeff_params = 0
    for name, m in model.named_modules():
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in pats):
            continue
        W = getattr(m, "weight", None)
        if not isinstance(W, torch.Tensor) or W.dim() != 2:
            continue
        # Ignore embeddings and layer norms
        if "embedding" in m.__class__.__name__.lower() or "norm" in m.__class__.__name__.lower():
            continue
        # Compute oriented dims
        W_o, b_o, in_f, out_f, _ = _orient_weight_bias(m)
        r = int(max(1, min(rank, min(out_f, in_f))))
        orig = out_f * in_f + (out_f if getattr(m, "bias", None) is not None else 0)
        coeff = r * (in_f + out_f + 1) + (out_f if getattr(m, "bias", None) is not None else 0)
        try:
            wrapper = SpectralLinearLike(m, rank=r)
            _set_by_name(model, name, wrapper)
            modules += 1
            orig_params += orig
            coeff_params += coeff
        except Exception:
            continue
    ratio = float(coeff_params) / max(1.0, float(orig_params))
    return {
        "modules": float(modules),
        "orig_params": float(orig_params),
        "coeff_params": float(coeff_params),
        "params_ratio": float(ratio),
        "rank": float(rank),
    }
