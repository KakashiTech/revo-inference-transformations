"""Tests for revo.spectral (spectral pruning)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.spectral import spectral_prune_tensor


class TestSpectral:
    def test_prune_keeps_shape(self):
        W = torch.randn(8, 8)
        Y, kept, total = spectral_prune_tensor(W, energy_keep=0.9)
        assert Y.shape == (8, 8)
        assert kept <= total

    def test_energy_1_keeps_all(self):
        W = torch.randn(8, 8)
        Y, kept, total = spectral_prune_tensor(W, energy_keep=1.0)
        assert kept == total

    def test_energy_0_removes_nearly_all(self):
        W = torch.randn(8, 8)
        Y, kept, total = spectral_prune_tensor(W, energy_keep=0.01)
        assert kept < total

    def test_low_energy_keeps_less(self):
        W = torch.randn(16, 16)
        _, kept_high, total = spectral_prune_tensor(W, energy_keep=0.95)
        _, kept_low, _ = spectral_prune_tensor(W, energy_keep=0.5)
        assert kept_low <= kept_high

    def test_higher_energy_keeps_more_coeffs(self):
        W = torch.randn(10, 10)
        _, kept_low, total = spectral_prune_tensor(W, energy_keep=0.5)
        _, kept_high, _ = spectral_prune_tensor(W, energy_keep=0.95)
        assert kept_high >= kept_low

    def test_non_square(self):
        W = torch.randn(6, 4)
        Y, kept, total = spectral_prune_tensor(W, energy_keep=0.9)
        assert Y.shape == (6, 4)

    def test_kept_ratio_between_zero_and_one(self):
        W = torch.randn(8, 8)
        for e in [0.1, 0.5, 0.9, 1.0]:
            Y, kept, total = spectral_prune_tensor(W, energy_keep=e)
            ratio = kept / total
            assert 0.0 <= ratio <= 1.0
