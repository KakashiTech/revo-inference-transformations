from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DeltaHandle:
    """Ephemeral delta handle."""
    delta: np.ndarray
    scale: float


def apply_delta(weight: np.ndarray, A: np.ndarray, B: np.ndarray, scale: float) -> Tuple[np.ndarray, DeltaHandle]:
    """Apply low-rank delta: new_weight = weight + (B@A)*scale, also return handle."""
    assert weight.ndim == 2, "weight must be 2D (out_features, in_features)"
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D"
    of, inf = weight.shape
    r, inf_a = A.shape
    of_b, r_b = B.shape
    assert inf_a == inf, "A in_features must match weight in_features"
    assert of_b == of and r_b == r, "B shape must match (out_features, rank)"

    delta = (B @ A) * float(scale)
    new_weight = weight + delta
    return new_weight, DeltaHandle(delta=delta, scale=float(scale))


def revert_delta(weight: np.ndarray, handle: DeltaHandle) -> np.ndarray:
    """Revert by subtracting stored delta (functional, no mutation)."""
    return weight - handle.delta
