from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules

def _orient_weight_bias(module: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    if not isinstance(W, torch.Tensor) or W.dim() != 2:
        raise AssertionError("module.weight must be 2D tensor")
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    W_det = W.detach()
    if b_det is not None:
        if W_det.shape[0] == b_det.numel():
            return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0])
        if W_det.shape[1] == b_det.numel():
            return W_det.t(), b_det, int(W_det.shape[0]), int(W_det.shape[1])
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0])


class FractalLinearWrap(nn.Module):
    """Self-similar composition of a linear mapping using powers of W.

    y = sum_{k=1..depth} beta_k * (W^k x) + b
    betas are parameters (frozen by default) initialized as geometric decay.
    """

    def __init__(self, base: nn.Module, depth: int = 2, alpha: float = 0.5, device=None, dtype=None):
        super().__init__()
        W_o, b_o, in_f, out_f = _orient_weight_bias(base)
        assert in_f == out_f, "FractalLinearWrap requires square mapping"
        self.features = int(in_f)
        self.depth = int(max(1, depth))
        factory_kwargs = {"device": device or W_o.device, "dtype": dtype or W_o.dtype}
        # Freeze oriented weights
        self.register_buffer("W", W_o.to(**factory_kwargs), persistent=False)
        self.register_buffer("b", (b_o.to(**factory_kwargs) if b_o is not None else None), persistent=False)
        # Coefficients beta_k with geometric decay alpha
        betas = [alpha ** (k - 1) for k in range(1, self.depth + 1)]
        self.register_buffer("beta", torch.tensor(betas, device=factory_kwargs["device"], dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_sum = torch.zeros(*x.shape[:-1], self.features, device=x.device, dtype=x.dtype)
        xk = x
        for k in range(self.depth):
            yk = F.linear(xk, self.W, None)
            y_sum = y_sum + self.beta[k] * yk
            xk = yk
        if self.b is not None:
            y_sum = y_sum + self.b
        return y_sum


@torch.no_grad()
def replace_with_fractal(
    model: nn.Module,
    depth: int = 2,
    alpha: float = 0.5,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
    allow_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    pats = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    report: Dict[str, int] = {}

    def set_by_name(root: nn.Module, path: str, new_mod: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else getattr(parent, "_modules")[p]
        last = parts[-1]
        if last.isdigit():
            parent._modules[last] = new_mod
        else:
            setattr(parent, last, new_mod)

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

    allow_set = set(allow_names) if allow_names is not None else None
    for name, m in model.named_modules():
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in pats):
            continue
        if allow_set is not None and name not in allow_set:
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
        # Require square mapping
        W_o, b_o, in_f, out_f = _orient_weight_bias(m)
        if in_f != out_f:
            continue
        wrapper = FractalLinearWrap(m, depth=depth, alpha=alpha, device=W.device, dtype=W.dtype)
        set_by_name(model, name, wrapper)
        report[name] = int(in_f)
    return report
