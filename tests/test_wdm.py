"""Tests for revo.wdm (WDMLinearWrap)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.wdm import WDMLinearWrap, _infer_in_out


class TestWDMLinearWrap:
    def test_forward_shape_square(self):
        base = nn.Linear(8, 8)
        w = WDMLinearWrap(base, bands=2)
        x = torch.randn(2, 8)
        y = w(x)
        assert y.shape == (2, 8)

    def test_requires_square(self):
        base = nn.Linear(8, 4)
        with pytest.raises(AssertionError):
            WDMLinearWrap(base, bands=2)

    def test_requires_divisible(self):
        base = nn.Linear(10, 10)
        with pytest.raises(AssertionError):
            WDMLinearWrap(base, bands=3)

    def test_bands_property(self):
        base = nn.Linear(12, 12)
        w = WDMLinearWrap(base, bands=3)
        assert w.bands == 3
        assert w.band_size == 4

    def test_forward_deterministic(self):
        base = nn.Linear(6, 6)
        w = WDMLinearWrap(base, bands=2)
        x = torch.randn(1, 6)
        y1 = w(x)
        y2 = w(x)
        assert torch.allclose(y1, y2)

    def test_forward_does_not_crash_batch(self):
        base = nn.Linear(16, 16)
        w = WDMLinearWrap(base, bands=4)
        x = torch.randn(3, 8, 16)
        y = w(x)
        assert y.shape == (3, 8, 16)

    def test_bias_preserved(self):
        base = nn.Linear(8, 8, bias=True)
        base.bias.data.fill_(0.5)
        w = WDMLinearWrap(base, bands=2)
        x = torch.zeros(1, 8)
        y = w(x)
        assert abs(y[0, 0].item() - 0.5) < 0.1

    def test_no_bias(self):
        base = nn.Linear(8, 8, bias=False)
        w = WDMLinearWrap(base, bands=2)
        assert w.bias is None

    def test_c_cols_band_size(self):
        base = nn.Linear(10, 10)
        w = WDMLinearWrap(base, bands=2)
        assert w.c_cols.shape == (2, 5)

    def test_infer_in_out(self):
        base = nn.Linear(8, 4)
        in_f, out_f = _infer_in_out(base)
        assert in_f == 8
        assert out_f == 4
