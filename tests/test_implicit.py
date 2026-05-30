"""Tests for revo.implicit (Phase VIII evaluation)."""
from __future__ import annotations

import torch
import pytest

from revo.implicit import ImplicitConfig, _make_prompts


class TestImplicit:
    def test_make_prompts_length(self):
        prompts = _make_prompts(4, seed=1)
        assert len(prompts) == 4

    def test_config_defaults(self):
        cfg = ImplicitConfig()
        assert cfg.codebook_k == 16
        assert cfg.q_quantile == 0.5

    def test_seed_deterministic(self):
        p1 = _make_prompts(3, seed=7)
        p2 = _make_prompts(3, seed=7)
        assert p1 == p2

    def test_different_seeds_differ(self):
        p1 = _make_prompts(5, seed=0)
        p2 = _make_prompts(5, seed=1)
        assert p1 != p2
