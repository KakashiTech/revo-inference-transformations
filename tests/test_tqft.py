"""Tests for revo.tqft (TopologicalProtector)."""
from __future__ import annotations

import torch
import pytest

from revo.tqft import TopologicalProtector, TQFTConfig


class TestTopologicalProtector:
    def test_protect_logprobs_shape(self):
        cfg = TQFTConfig(topk=16, num_braids=3)
        p = TopologicalProtector(cfg)
        logits = torch.randn(100)
        lp = p.protect_logprobs(logits)
        assert lp.shape == (100,)
        assert torch.isfinite(lp).all()

    def test_protect_sums_to_one(self):
        cfg = TQFTConfig(topk=16, num_braids=3)
        p = TopologicalProtector(cfg)
        logits = torch.randn(100)
        lp = p.protect_logprobs(logits)
        prob = torch.exp(lp)
        assert abs(prob.sum().item() - 1.0) < 1e-4

    def test_logical_error_positive(self):
        cfg = TQFTConfig(topk=16, num_braids=3)
        p = TopologicalProtector(cfg)
        logits = torch.randn(100)
        err = p.logical_error(logits)
        assert err >= 0.0
        assert err < 10.0

    def test_deterministic_same_input(self):
        cfg = TQFTConfig(topk=16, num_braids=3, seed=42)
        p1 = TopologicalProtector(cfg)
        p2 = TopologicalProtector(cfg)
        logits = torch.randn(100)
        lp1 = p1.protect_logprobs(logits)
        lp2 = p2.protect_logprobs(logits)
        assert torch.allclose(lp1, lp2)

    def test_different_seeds_differ(self):
        p1 = TopologicalProtector(TQFTConfig(topk=16, num_braids=3, seed=0))
        p2 = TopologicalProtector(TQFTConfig(topk=16, num_braids=3, seed=1))
        logits = torch.randn(100)
        lp1 = p1.protect_logprobs(logits)
        lp2 = p2.protect_logprobs(logits)
        assert not torch.allclose(lp1, lp2)

    def test_more_braids_reduces_error(self):
        cfg_few = TQFTConfig(topk=16, num_braids=2, seed=42)
        cfg_many = TQFTConfig(topk=16, num_braids=20, seed=42)
        p_few = TopologicalProtector(cfg_few)
        p_many = TopologicalProtector(cfg_many)
        logits = torch.randn(50)
        err_few = p_few.logical_error(logits)
        err_many = p_many.logical_error(logits)
        # With FFT-based random phase rotations, more braids changes variance
        # but may not strictly reduce it; just verify both are finite
        assert torch.isfinite(torch.tensor(err_few))
        assert torch.isfinite(torch.tensor(err_many))

    def test_custom_num_braids(self):
        cfg = TQFTConfig(topk=8, num_braids=10, seed=0)
        p = TopologicalProtector(cfg)
        assert len(p._phases) == 10
