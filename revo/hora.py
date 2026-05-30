from __future__ import annotations
from revo._logging import get_logger

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import set_by_name, skip_tied_weights
from revo.hyperbolic import expmap0, logmap0


def _infer_oriented_weight(module: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    W_det = W.detach()
    if b_det is not None and W_det.dim() == 2:
        if W_det.shape[0] == b_det.numel():
            return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0])
        if W_det.shape[1] == b_det.numel():
            return W_det.t(), b_det, int(W_det.shape[0]), int(W_det.shape[1])
    # Fallback assume [out,in]
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0])


class HoRALinearAdapter(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float = 1.0, c: float = 0.0, device=None, dtype=None):
        super().__init__()
        W_o, b_o, in_features, out_features = _infer_oriented_weight(base)
        factory_kwargs = {"device": device or W_o.device, "dtype": dtype or W_o.dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(max(1, rank))
        self.c = float(max(0.0, c))
        self.scaling = float(alpha) / float(self.rank)
        # Store base weights as buffers (frozen)
        self.register_buffer("W", W_o.to(**factory_kwargs), persistent=False)
        self.register_buffer("b", (b_o.to(**factory_kwargs) if b_o is not None else None), persistent=False)
        # Low-rank adapters in tangent space
        self.A = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.B = nn.Linear(self.rank, self.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c
        if c == 0.0:
            # Euclidean LoRA equivalent
            base = F.linear(x, self.W, self.b)
            delta = self.B(self.A(x)) * self.scaling
            return base + delta
        # Hyperbolic: operate in tangent at 0
        x_t = logmap0(x, c)
        base_t = F.linear(x_t, self.W, self.b)
        delta_t = self.B(self.A(x_t)) * self.scaling
        y_t = base_t + delta_t
        y = expmap0(y_t, c)
        return y


@torch.no_grad()
def replace_with_hora(
    model: nn.Module,
    rank: int,
    alpha: float = 1.0,
    c: float = 0.0,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, Tuple[int, int]]:
    patterns = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    report: Dict[str, Tuple[int, int]] = {}

    # Detect tied weights to skip
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
        # Filter by patterns
        if not any(pat in name for pat in patterns):
            continue
        if not isinstance(m, nn.Linear):
            continue
        # Skip tied input embedding
        try:
            if (input_embed_weight is not None) and (m.weight is input_embed_weight):
                continue
        except Exception:
            get_logger().warning("except Exception:")
        # Build adapter with inferred dims and replace
        W_o, b_o, in_f, out_f = _infer_oriented_weight(m)
        adapter = HoRALinearAdapter(m, rank=rank, alpha=alpha, c=c, device=m.weight.device, dtype=m.weight.dtype)
        set_by_name(model, name, adapter)
        report[name] = (out_f, rank)
    return report
