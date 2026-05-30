"""Tests for revo.reversible (ReversibleUncomputeWrap)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.reversible import ReversibleUncomputeWrap


class TestReversibleUncomputeWrap:
    def test_forward_shape(self):
        base = nn.Linear(8, 8)
        r = ReversibleUncomputeWrap(base, rank=2)
        x = torch.randn(2, 8)
        y = r(x)
        assert y.shape == (2, 8)

    def test_gamma_exists(self):
        base = nn.Linear(8, 8)
        r = ReversibleUncomputeWrap(base, rank=2)
        assert hasattr(r, "gamma")
        assert r.gamma.numel() == 1

    def test_gamma_init_close_to_zero(self):
        base = nn.Linear(8, 8)
        r = ReversibleUncomputeWrap(base, rank=2)
        # sigmoid(-9) ~ 0.0001, so effect is very small
        beta = torch.sigmoid(r.gamma).item()
        assert beta < 0.01

    def test_forward_close_to_base_when_gamma_small(self):
        base = nn.Linear(8, 8)
        nn.init.eye_(base.weight)
        base.bias.data.zero_()
        r = ReversibleUncomputeWrap(base, rank=2)
        x = torch.randn(1, 8)
        y_base = base(x)
        y_rev = r(x)
        diff = (y_rev - y_base).abs().mean().item()
        assert diff < 1.0

    def test_rank_clamp(self):
        base = nn.Linear(4, 4)
        r = ReversibleUncomputeWrap(base, rank=100)
        assert r.rank <= 4

    def test_ab_shapes(self):
        base = nn.Linear(8, 8)
        r = ReversibleUncomputeWrap(base, rank=3)
        assert r.A.weight.shape == (3, 8)
        assert r.B.weight.shape == (8, 3)

    def test_output_is_finite(self):
        base = nn.Linear(8, 4)
        r = ReversibleUncomputeWrap(base, rank=2)
        x = torch.randn(2, 8)
        y = r(x)
        assert torch.isfinite(y).all()
