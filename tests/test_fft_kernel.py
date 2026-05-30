"""Tests for revo.fft_kernel (CirculantLinear)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.fft_kernel import CirculantLinear, nearest_circulant_first_column


class TestCirculantLinear:
    def test_forward_shape(self):
        c = CirculantLinear(8, bias=True)
        c.c.copy_(torch.randn(8))
        x = torch.randn(2, 8)
        y = c(x)
        assert y.shape == (2, 8)

    def test_forward_deterministic(self):
        c = CirculantLinear(6, bias=False)
        c.c.copy_(torch.randn(6))
        x = torch.randn(1, 6)
        y1 = c(x)
        y2 = c(x)
        assert torch.allclose(y1, y2)

    def test_with_bias(self):
        c = CirculantLinear(8, bias=True)
        c.c.copy_(torch.randn(8))
        c.bias.data.fill_(0.5)
        x = torch.zeros(1, 8)
        y = c(x)
        assert abs(y[0, 0].item() - 0.5) < 1e-6

    def test_no_bias(self):
        c = CirculantLinear(8, bias=False)
        assert c.bias is None

    def test_nearest_circulant_first_column_shape(self):
        W = torch.randn(8, 8)
        c = nearest_circulant_first_column(W)
        assert c.shape == (8,)

    def test_circulant_from_diagonal(self):
        W = torch.eye(6)
        c = nearest_circulant_first_column(W)
        # For identity, nearest circulant is (1/n) * ones(n)
        expected = torch.ones(6) / 6.0
        assert torch.allclose(c, expected, atol=1e-6)

    def test_set_from_weight(self):
        c = CirculantLinear(8, bias=True)
        W = torch.randn(8, 8)
        b = torch.randn(8)
        c.set_from_weight(W, b)
        assert torch.isfinite(c.c).all()
        assert torch.allclose(c.bias, b, atol=1e-6)

    def test_batch_forward(self):
        c = CirculantLinear(6, bias=False)
        c.c.copy_(torch.randn(6))
        x = torch.randn(2, 3, 6)
        y = c(x)
        assert y.shape == (2, 3, 6)
