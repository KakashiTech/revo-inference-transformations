from __future__ import annotations
from revo._logging import get_logger

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules

class ResonanceGateWrap(nn.Module):
    """Gated wrapper: y = sigma(gamma) * base(x).

    Minimal, stable gating to emulate ephemeral activation conditioned by context.
    A calibration routine can set gamma close to 1 to preserve accuracy.
    """

    def __init__(self, base: nn.Module, device=None, dtype=None, gamma_init: float | None = None):
        super().__init__()
        self.base = base
        # Start close to identity (sigma(2.0) ~ 0.88)
        g0 = 2.0 if gamma_init is None else float(gamma_init)
        self.gamma = nn.Parameter(torch.tensor(g0, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        gate = torch.sigmoid(self.gamma)
        return gate * y


@torch.no_grad()
def replace_with_ephemeral(
    model: nn.Module,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
    allow_names: Optional[List[str]] = None,
    gamma_init_map: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    pats = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    report: Dict[str, float] = {}

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

    # Skip tied head
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
        # Only wrap modules with 2D weight to stay in linear-like layers
        if not hasattr(m, "weight") or not isinstance(getattr(m, "weight"), torch.Tensor) or getattr(m, "weight").dim() != 2:
            continue
        try:
            if (input_embed_weight is not None) and (m.weight is input_embed_weight):
                continue
        except Exception:
            get_logger().warning("except Exception:")
        g_init = gamma_init_map.get(name) if gamma_init_map is not None else None
        wrapper = ResonanceGateWrap(m, device=m.weight.device, dtype=m.weight.dtype, gamma_init=g_init)
        set_by_name(model, name, wrapper)
        report[name] = 1.0
    return report


def calibrate_ephemeral(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    steps: int = 10,
    lr: float = 5e-2,
    lambda_phys: float = 1.0,
    max_length: int = 128,
) -> Dict[str, float]:
    # Train only gamma parameters
    params = []
    for m in model.modules():
        if isinstance(m, ResonanceGateWrap):
            params.append(m.gamma)
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
            # Smoothness in time as physical prior
            d1 = logits[:, 1:, :] - logits[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            loss_phys = (d2 * d2).mean()
            loss_total = loss_total + lambda_phys * loss_phys
        loss_total.backward()
        opt.step()
    model.eval()
    return {"updated": float(len(params))}
