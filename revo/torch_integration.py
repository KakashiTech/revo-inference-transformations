from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class TorchDeltaHandle:
    module: torch.nn.Module
    delta: torch.Tensor


def apply_delta_torch(module: torch.nn.Module, A: np.ndarray, B: np.ndarray, scale: float) -> TorchDeltaHandle:
    """Apply low-rank delta to a torch module with 2D weight, return handle for revert."""
    w = module.weight
    assert w.dim() == 2, "module.weight must be 2D (out_features, in_features)"
    of, inf = w.shape
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
    r, inf_a = A.shape
    of_b, r_b = B.shape
    assert inf_a == inf, "A in_features must match module in_features"
    assert of_b == of and r_b == r, "B shape must match (out_features, rank)"

    device = w.device
    dtype = w.dtype
    A_t = torch.from_numpy(A).to(device=device, dtype=dtype)
    B_t = torch.from_numpy(B).to(device=device, dtype=dtype)
    delta = torch.matmul(B_t, A_t) * float(scale)

    with torch.no_grad():
        w.add_(delta)

    return TorchDeltaHandle(module=module, delta=delta)


def revert_delta_torch(handle: TorchDeltaHandle) -> None:
    """Revert previously applied delta (functional revert)."""
    with torch.no_grad():
        handle.module.weight.sub_(handle.delta)
