"""Tests for revo.regimes (Phase VI)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.regimes import RegimeConfig, _n_params, _smooth_activation, _build_gamma_map


class TestRegimes:
    def test_config_defaults(self):
        cfg = RegimeConfig()
        assert cfg.density_weight == 0.5
        assert cfg.depth_weight == 0.5
        assert cfg.center == 0.6
        assert cfg.sharpness == 4.0

    def test_n_params(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        n = _n_params(model)
        assert n > 0

    def test_smooth_activation_zero(self):
        val = _smooth_activation(0.0, center=0.5, sharpness=10.0)
        assert val < 0.01

    def test_smooth_activation_one(self):
        val = _smooth_activation(1.0, center=0.5, sharpness=10.0)
        assert val > 0.99

    def test_smooth_activation_center(self):
        val = _smooth_activation(0.5, center=0.5, sharpness=4.0)
        assert abs(val - 0.5) < 0.1

    def test_smooth_activation_symmetric(self):
        v1 = _smooth_activation(0.3, center=0.5, sharpness=4.0)
        v2 = _smooth_activation(0.7, center=0.5, sharpness=4.0)
        assert abs(v1 + v2 - 1.0) < 0.1

    def test_build_gamma_map(self):
        model = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )
        gamma_map = _build_gamma_map(model, 0.5, ("0", "1"))
        assert len(gamma_map) >= 1
        for v in gamma_map.values():
            assert 0.0 <= v <= 1.0
