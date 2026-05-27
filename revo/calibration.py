from __future__ import annotations
from revo._logging import get_logger

from typing import List, Optional

import torch
import torch.nn as nn


class LogitCalibWrap(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y * self.scale + self.bias


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


def wrap_lm_head_with_calib(model: nn.Module) -> Optional[str]:
    # prefer explicit lm_head
    for name, m in model.named_modules():
        if name == "lm_head" and isinstance(m, nn.Module):
            _set_by_name(model, name, LogitCalibWrap(m))
            return name
    # fallback via get_output_embeddings
    try:
        get_head = getattr(model, "get_output_embeddings", None)
        if callable(get_head):
            head = get_head()
            # find its path
            for name, m in model.named_modules():
                if m is head:
                    _set_by_name(model, name, LogitCalibWrap(m))
                    return name
    except Exception:
        get_logger().warning("except Exception:")
    return None


def calibrate_logits(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    steps: int = 50,
    lr: float = 1e-2,
    max_length: int = 128,
) -> None:
    # enable grad only for LogitCalibWrap params
    params = []
    for m in model.modules():
        if isinstance(m, LogitCalibWrap):
            params.extend([m.scale, m.bias])
    if not params:
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    device = next(model.parameters()).device
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])  # type: ignore[arg-type]
            loss_total = loss_total + out.loss
        (loss_total / max(1, len(texts))).backward()
        opt.step()
    model.eval()
