"""Tests for revo.holography (HoloBoundaryAdapter)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.holography import HoloBoundaryAdapter, _orient_weight_bias


class TestHoloBoundaryAdapter:
    def test_forward_shape(self):
        base = nn.Linear(16, 8)
        adapter = HoloBoundaryAdapter(base, boundary_dim=4)
        x = torch.randn(2, 16)
        y = adapter(x)
        assert y.shape == (2, 8)

    def test_boundary_dim_clamp(self):
        base = nn.Linear(4, 4)
        adapter = HoloBoundaryAdapter(base, boundary_dim=100)
        assert adapter.boundary_dim <= 4

    def test_gamma_init(self):
        base = nn.Linear(8, 8)
        adapter = HoloBoundaryAdapter(base, boundary_dim=3)
        assert hasattr(adapter, "gamma")
        assert adapter.gamma.numel() == 1

    def test_perturbation_is_small_initial(self):
        base = nn.Linear(8, 8)
        nn.init.eye_(base.weight)
        base.bias.data.zero_()
        adapter = HoloBoundaryAdapter(base, boundary_dim=3)
        x = torch.randn(1, 8)
        y_base = torch.nn.functional.linear(x, base.weight, base.bias)
        y_adapted = adapter(x)
        diff = (y_adapted - y_base).abs().mean().item()
        assert diff < 1.0  # small perturbation due to sigmoid(gamma)~0.5

    def test_q_orthonormal(self):
        base = nn.Linear(12, 12)
        adapter = HoloBoundaryAdapter(base, boundary_dim=5)
        QtQ = adapter.Q.t() @ adapter.Q
        I = torch.eye(5)
        assert torch.allclose(QtQ, I, atol=1e-6)

    def test_orients_weight_bias(self):
        base = nn.Linear(10, 6)
        W, b, in_f, out_f = _orient_weight_bias(base)
        assert in_f == 10
        assert out_f == 6
        assert W.shape == (6, 10)

    def test_no_bias(self):
        base = nn.Linear(8, 8, bias=False)
        adapter = HoloBoundaryAdapter(base, boundary_dim=2)
        assert adapter.b is None
        x = torch.randn(1, 8)
        y = adapter(x)
        assert y.shape == (1, 8)

    def test_different_alpha(self):
        base = nn.Linear(8, 8)
        adapter = HoloBoundaryAdapter(base, boundary_dim=2, alpha=2.0)
        assert adapter.alpha == 2.0
