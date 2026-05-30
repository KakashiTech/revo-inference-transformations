"""Tests for revo.holomorphic (HolomorphicFourierLinearLike)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.holomorphic import HolomorphicFourierLinearLike


class TestHolomorphicFourierLinearLike:
    def test_forward_shape(self):
        base = nn.Linear(16, 8)
        h = HolomorphicFourierLinearLike(base, rank=4)
        x = torch.randn(2, 16)
        y = h(x)
        assert y.shape == (2, 8)

    def test_fewer_params_than_dense(self):
        base = nn.Linear(64, 32)
        h = HolomorphicFourierLinearLike(base, rank=4)
        dense_params = 64 * 32 + 32  # weight + bias
        holo_params = h.rank + h.in_features + h.out_features  # coeffs + a_in + a_out
        if h.bias is not None:
            holo_params += h.out_features
        assert holo_params < dense_params

    def test_forward_deterministic(self):
        base = nn.Linear(8, 8)
        h = HolomorphicFourierLinearLike(base, rank=3)
        x = torch.randn(1, 8)
        y1 = h(x)
        y2 = h(x)
        assert torch.allclose(y1, y2)

    def test_coeffs_initialized(self):
        base = nn.Linear(8, 4)
        h = HolomorphicFourierLinearLike(base, rank=3)
        assert h.coeffs.shape == (3,)
        assert torch.isfinite(h.coeffs).all()

    def test_a_in_a_out_present(self):
        base = nn.Linear(8, 4)
        h = HolomorphicFourierLinearLike(base, rank=2)
        assert h.a_in.shape == (8,)
        assert h.a_out.shape == (4,)

    def test_rank_clamp(self):
        base = nn.Linear(4, 4)
        h = HolomorphicFourierLinearLike(base, rank=100)
        assert h.rank <= 4

    def test_bias_none(self):
        base = nn.Linear(8, 4, bias=False)
        h = HolomorphicFourierLinearLike(base, rank=2)
        assert h.bias is None

    def test_output_finite(self):
        base = nn.Linear(10, 6)
        h = HolomorphicFourierLinearLike(base, rank=3)
        x = torch.randn(3, 10)
        y = h(x)
        assert torch.isfinite(y).all()
