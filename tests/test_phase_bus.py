"""Tests for revo.phase_bus (PhaseBusWrap)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.phase_bus import PhaseBusWrap, _rfft_phase_apply


class TestPhaseBusWrap:
    def test_forward_shape(self):
        base = nn.Linear(8, 4)
        w = PhaseBusWrap(base, in_features=8, out_features=4)
        x = torch.randn(2, 8)
        y = w(x)
        assert y.shape == (2, 4)

    def test_theta_params(self):
        base = nn.Linear(6, 6)
        w = PhaseBusWrap(base, in_features=6, out_features=6)
        assert hasattr(w, "theta_in")
        assert hasattr(w, "theta_out")
        assert w.theta_in.shape == (4,)  # 6//2+1
        assert w.theta_out.shape == (4,)

    def test_rfft_apply_shape(self):
        x = torch.randn(2, 8)
        theta = torch.zeros(5)  # 8//2+1
        y = _rfft_phase_apply(x, theta)
        assert y.shape == (2, 8)

    def test_zero_theta_identity(self):
        x = torch.randn(2, 8)
        theta = torch.zeros(5)
        y = _rfft_phase_apply(x, theta)
        assert torch.allclose(x, y, atol=1e-6)

    def test_nonzero_theta_changes(self):
        x = torch.randn(2, 8)
        theta = torch.ones(5) * 0.5
        y = _rfft_phase_apply(x, theta)
        assert not torch.allclose(x, y, atol=1e-4)

    def test_forward_deterministic(self):
        base = nn.Linear(4, 4)
        w = PhaseBusWrap(base, in_features=4, out_features=4)
        x = torch.randn(1, 4)
        y1 = w(x)
        y2 = w(x)
        assert torch.allclose(y1, y2)
