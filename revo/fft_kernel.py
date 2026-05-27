from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules


def _infer_in_out_from_weight(module: nn.Module) -> Tuple[int, int]:
    assert hasattr(module, "weight"), "module must have .weight"
    W = getattr(module, "weight")
    if not isinstance(W, torch.Tensor) or W.dim() != 2:
        raise AssertionError("module.weight must be 2D tensor")
    b = getattr(module, "bias", None)
    if isinstance(b, torch.Tensor):
        if W.shape[0] == b.numel():
            out_f, in_f = int(W.shape[0]), int(W.shape[1])
        elif W.shape[1] == b.numel():
            out_f, in_f = int(W.shape[1]), int(W.shape[0])
        else:
            out_f, in_f = int(W.shape[0]), int(W.shape[1])
    else:
        out_f, in_f = int(W.shape[0]), int(W.shape[1])
    return in_f, out_f


@torch.no_grad()
def nearest_circulant_first_column(W: torch.Tensor) -> torch.Tensor:
    """Return the first column c of the Frobenius-nearest circulant to W (square).

    c[k] = (1/n) * sum_i W[i, (i-k) mod n]
    """
    assert W.dim() == 2 and W.shape[0] == W.shape[1]
    n = W.shape[0]
    c = torch.fft.irfft(torch.fft.rfft(W, dim=-1).mean(dim=0), n=n).to(dtype=W.dtype, device=W.device)
    return c


class CirculantLinear(nn.Module):
    """y = Cx + b using FFTs, where C is circulant with first column c.

    Only supports in_features == out_features.
    """

    def __init__(self, features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.features = int(features)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.register_buffer("c", torch.zeros(self.features, **factory_kwargs), persistent=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def set_from_weight(self, W: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        assert W.dim() == 2 and W.shape[0] == W.shape[1] == self.features
        c = nearest_circulant_first_column(W)
        self.c.copy_(c)
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., N]
        N = self.features
        assert x.size(-1) == N
        # Compute y = ifft(fft(c) * fft(x))
        c_freq = torch.fft.rfft(self.c)  # [N//2+1]
        x_freq = torch.fft.rfft(x, dim=-1)
        y_freq = x_freq * c_freq
        y = torch.fft.irfft(y_freq, n=N, dim=-1)
        if self.bias is not None:
            y = y + self.bias
        return y


@torch.no_grad()
def replace_with_circulant(
    model: nn.Module,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
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

    # Skip tied weights
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
        # Infer in/out and require square effective mapping
        in_f, out_f = _infer_in_out_from_weight(m)
        if in_f != out_f:
            continue
        N = in_f
        device = W.device
        dtype = W.dtype
        bias = getattr(m, "bias", None)
        bias_t = bias.detach().to(device=device, dtype=dtype) if isinstance(bias, torch.Tensor) else None
        wrapper = CirculantLinear(N, bias=(bias_t is not None), device=device, dtype=dtype)
        # Orient W to [out,in] consistent with bias
        if isinstance(bias, torch.Tensor) and W.shape[1] == bias.numel():
            W_use = W.t()
        else:
            W_use = W
        wrapper.set_from_weight(W_use, bias=bias_t)
        set_by_name(model, name, wrapper)
        report[name] = int(N)
    return report
