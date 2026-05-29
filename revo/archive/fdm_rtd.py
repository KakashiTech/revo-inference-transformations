from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export all WDM functionality for backward compatibility
from revo.wdm import WDMLinearWrap, replace_with_wdm

__all__ = [
    "WDMLinearWrap",
    "replace_with_wdm",
    "FDMFilterBank",
]


class FDMFilterBank(nn.Module):
    """I.4 Tesla Resonance Routing – frequency-division multiplexed filter bank.

    Splits the input feature space into ``n_bands`` independent sub-channels
    (frequency bands) via learned linear projections (bandpass filters).
    Each band is processed by a dedicated small linear layer, then the bands
    are concatenated and recombined into the output space.

    Physics-inspired name mapping:
        FDM (Frequency-Division Multiplexing)  → learned per-band projections
        RTD (Resonant Tunnelling Diode)        → per-band non-linear gating
    """

    def __init__(
        self,
        n_bands: int,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        assert in_features % n_bands == 0, \
            f"in_features ({in_features}) must be divisible by n_bands ({n_bands})"
        self.n_bands = n_bands
        self.in_features = in_features
        self.out_features = out_features
        self.band_size = in_features // n_bands

        # Learned bandpass filters  [n_bands, in_features, band_size]
        self.bandpass = nn.Parameter(
            torch.randn(n_bands, in_features, self.band_size) * 0.02
        )

        # Per-band linear processors + RTD-style gating  (band_size -> band_size)
        self.band_linears = nn.ModuleList()
        self.band_gates = nn.ModuleList()
        for _ in range(n_bands):
            self.band_linears.append(nn.Linear(self.band_size, self.band_size))
            self.band_gates.append(nn.Linear(self.band_size, self.band_size))

        # Recombination projection
        self.recombine = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: filter → process → recombine.

        Args:
            x: Tensor of shape ``(..., in_features)``.

        Returns:
            Tensor of shape ``(..., out_features)``.
        """
        *dims, N = x.shape
        # 1) Bandpass filtering
        bands: List[torch.Tensor] = []
        for i in range(self.n_bands):
            # x · W_i  →  [*, band_size]
            band = torch.matmul(x, self.bandpass[i])
            # 2) Independent linear processing  +  RTD sigmoid gate
            lin = self.band_linears[i](band)
            gate = torch.sigmoid(self.band_gates[i](band))
            band = lin * gate
            bands.append(band)

        # 3) Concatenate bands  →  [*, in_features]
        fused = torch.cat(bands, dim=-1)

        # 4) Recombine  →  [*, out_features]
        return self.recombine(fused)
