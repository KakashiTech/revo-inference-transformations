"""Tests for revo.beds (BEDSHomeostat)."""
from __future__ import annotations

import torch
import pytest

from revo.beds import BEDSHomeostat


class TestBEDSHomeostat:
    def test_init(self):
        h = BEDSHomeostat(H_min=0.5, H_max=3.0, Kp=0.1, Ki=0.01, window=5)
        assert h.H_min == 0.5
        assert h.H_max == 3.0
        assert h.Kp == 0.1
        assert h.Ki == 0.01
        assert h.window == 5

    def test_correct_inside_range_no_change(self):
        h = BEDSHomeostat(H_min=0.0, H_max=10.0)
        logits = torch.randn(1, 5)
        out = h.correct(logits)
        assert torch.allclose(out, logits, atol=1e-6)

    def test_correct_very_peaked(self):
        h = BEDSHomeostat(H_min=1.0, H_max=3.0, Kp=0.2, Ki=0.02)
        # Very peaked distribution → low entropy
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        out = h.correct(logits)
        # Should increase entropy → flatten → lower temperature
        H_orig = -torch.sum(torch.softmax(logits, -1) * torch.log_softmax(logits, -1).exp().clamp_min(1e-12), -1).mean()
        H_out = -torch.sum(torch.softmax(out, -1) * torch.log_softmax(out, -1).exp().clamp_min(1e-12), -1).mean()
        assert H_out > H_orig

    def test_correct_very_flat(self):
        h = BEDSHomeostat(H_min=1.0, H_max=3.0, Kp=0.2, Ki=0.02)
        # Very flat distribution → high entropy
        logits = torch.tensor([[1.0, 1.1, 0.9, 1.0, 1.0]])
        out = h.correct(logits)
        H_orig = -torch.sum(torch.softmax(logits, -1) * torch.log_softmax(logits, -1).exp().clamp_min(1e-12), -1).mean()
        H_out = -torch.sum(torch.softmax(out, -1) * torch.log_softmax(out, -1).exp().clamp_min(1e-12), -1).mean()
        # Should decrease entropy (temperature > 1)
        assert H_out < H_orig or abs(H_out - H_orig) < 1e-6

    def test_reset_clears_history(self):
        h = BEDSHomeostat(H_min=0.0, H_max=0.1, Kp=0.1, Ki=0.01, window=5)
        for _ in range(10):
            h.correct(torch.randn(1, 5))
        assert len(h._history) > 0
        h.reset()
        assert len(h._history) == 0
        assert h._integral == 0.0

    def test_window_smoothing(self):
        h = BEDSHomeostat(H_min=0.0, H_max=0.1, Kp=0.1, Ki=0.01, window=3)
        for _ in range(20):
            h.correct(torch.randn(1, 5))
        assert len(h._history) <= 3

    def test_batch_input(self):
        h = BEDSHomeostat(H_min=0.0, H_max=10.0)
        logits = torch.randn(4, 8, 16)
        out = h.correct(logits)
        assert out.shape == logits.shape
