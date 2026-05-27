from __future__ import annotations
from revo._logging import get_logger

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules

def _rfft_phase_apply(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    # x: [..., N], theta: [N//2+1]
    N = x.size(-1)
    Xf = torch.fft.rfft(x, dim=-1)
    phase = torch.exp(1j * theta)
    Yf = Xf * phase
    y = torch.fft.irfft(Yf, n=N, dim=-1)
    return y


class PhaseBusWrap(nn.Module):
    """Wrapper that applies frequency-phase alignment before and after a base linear-like module.

    Pre: x' = irfft(rfft(x) * e^{i theta_in}); Post: y' = irfft(rfft(y) * e^{i theta_out}).
    The phase parameters are unit-modulus via theta ∈ R.
    """

    def __init__(self, base: nn.Module, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.base = base
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.theta_in = nn.Parameter(torch.zeros(self.in_features // 2 + 1, device=device))
        self.theta_out = nn.Parameter(torch.zeros(self.out_features // 2 + 1, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = _rfft_phase_apply(x, self.theta_in)
        y = self.base(x2)
        y2 = _rfft_phase_apply(y, self.theta_out)
        return y2


@torch.no_grad()
def _infer_in_out(module: nn.Module) -> Tuple[int, int]:
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    if isinstance(b, torch.Tensor):
        if W.shape[0] == b.numel():
            return int(W.shape[1]), int(W.shape[0])
        if W.shape[1] == b.numel():
            return int(W.shape[0]), int(W.shape[1])
    return int(W.shape[1]), int(W.shape[0])


@torch.no_grad()
def replace_with_phase_bus(
    model: nn.Module,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, Tuple[int, int]]:
    pats = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    report: Dict[str, Tuple[int, int]] = {}

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
            get_logger().warning("except Exception:")
        in_f, out_f = _infer_in_out(m)
        wrapper = PhaseBusWrap(m, in_features=in_f, out_features=out_f, device=W.device, dtype=W.dtype)
        set_by_name(model, name, wrapper)
        report[name] = (in_f, out_f)
    return report


def calibrate_phase_bus(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    steps: int = 10,
    lr: float = 5e-2,
    lambda_phys: float = 1.0,
    max_length: int = 128,
) -> Dict[str, float]:
    # Enable grads only for theta params
    params = []
    for m in model.modules():
        if isinstance(m, PhaseBusWrap):
            params.extend([m.theta_in, m.theta_out])
    if not params:
        return {"updated": 0.0}
    for p in model.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            logits = out.logits
            d1 = logits[:, 1:, :] - logits[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            loss_phys = (d2 * d2).mean()
            loss_total = loss_total + lambda_phys * loss_phys
        loss_total.backward()
        opt.step()
    model.eval()
    return {"updated": float(len(params))}
