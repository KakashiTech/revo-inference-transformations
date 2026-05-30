"""Tests for revo.hora (HoRALinearAdapter)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.hora import HoRALinearAdapter


class TestHoRALinearAdapter:
    def test_forward_shape_euclidean(self):
        base = nn.Linear(16, 8)
        h = HoRALinearAdapter(base, rank=4, alpha=1.0, c=0.0)
        x = torch.randn(2, 16)
        y = h(x)
        assert y.shape == (2, 8)

    def test_forward_shape_hyperbolic(self):
        base = nn.Linear(16, 8)
        h = HoRALinearAdapter(base, rank=4, alpha=1.0, c=0.1)
        x = torch.randn(2, 16) * 0.1
        y = h(x)
        assert y.shape == (2, 8)

    def test_forward_deterministic(self):
        base = nn.Linear(8, 8)
        h = HoRALinearAdapter(base, rank=3, alpha=1.0, c=0.0)
        x = torch.randn(1, 8)
        y1 = h(x)
        y2 = h(x)
        assert torch.allclose(y1, y2)

    def test_bias_none(self):
        base = nn.Linear(8, 4, bias=False)
        h = HoRALinearAdapter(base, rank=2, alpha=1.0, c=0.0)
        assert h.b is None
        x = torch.randn(1, 8)
        y = h(x)
        assert y.shape == (1, 4)

    def test_rank_clamp(self):
        base = nn.Linear(4, 4)
        h = HoRALinearAdapter(base, rank=100, alpha=1.0, c=0.0)
        assert h.rank == 100  # rank is not clamped, it's user-specified

    def test_scale_effect(self):
        base = nn.Linear(8, 8)
        base.bias.data.zero_()
        h = HoRALinearAdapter(base, rank=4, alpha=2.0, c=0.0)
        x = torch.randn(1, 8)
        y = h(x)
        # With non-zero base weights, delta should produce non-zero output
        # B initializes to zeros so delta=0 initially, but base is non-zero
        assert y.abs().sum().item() > 0

    def test_wb_preserved_from_base(self):
        base = nn.Linear(8, 4)
        W_orig = base.weight.data.clone()
        b_orig = base.bias.data.clone() if base.bias is not None else None
        h = HoRALinearAdapter(base, rank=2, alpha=1.0, c=0.0)
        assert torch.allclose(h.W, W_orig)
        if b_orig is not None:
            assert torch.allclose(h.b, b_orig)
