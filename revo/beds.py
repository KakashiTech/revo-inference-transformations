"""BEDS — Bayesian Emergent Dissipative Structures.

Implements II.1 from GRH.md: a minimal proxy that models inference as an
entropy-exporting dissipative process.  Tracks semantic entropy per token and
applies a homeostatic correction to log-probabilities when entropy deviates
from a target range.

CPU-only, pure PyTorch.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class BEDSHomeostat:
    """Homeostatic controller that nudges logits toward target entropy.

    For each token, computes H = -sum(p * log p) of the softmax output.
    If H drifts outside [H_min, H_max], applies a small corrective temperature
    shift proportional to the deviation.
    """

    def __init__(
        self,
        H_min: float = 0.5,
        H_max: float = 3.5,
        Kp: float = 0.05,
        Ki: float = 0.01,
        window: int = 10,
    ) -> None:
        self.H_min = float(H_min)
        self.H_max = float(H_max)
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.window = int(max(1, window))
        self._integral: float = 0.0
        self._history: List[float] = []

    def reset(self) -> None:
        self._integral = 0.0
        self._history.clear()

    def _entropy(self, logits: torch.Tensor) -> float:
        p = F.softmax(logits, dim=-1)
        h = -torch.sum(p * torch.log(p.clamp(min=1e-12)), dim=-1)
        return float(h.mean().item())

    def correct(self, logits: torch.Tensor) -> torch.Tensor:
        H = self._entropy(logits)
        self._history.append(H)
        if len(self._history) > self.window:
            self._history.pop(0)
        H_smooth = float(sum(self._history)) / max(1, len(self._history))

        if H_smooth < self.H_min:
            error = self.H_min - H_smooth
        elif H_smooth > self.H_max:
            error = self.H_max - H_smooth
        else:
            error = 0.0

        self._integral += error

        temp_adj = self.Kp * error + self.Ki * self._integral
        temp_adj = max(-0.5, min(0.5, temp_adj))

        if abs(temp_adj) < 1e-8:
            return logits
        tau = math.exp(temp_adj)
        return logits / tau
