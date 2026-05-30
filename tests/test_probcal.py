"""Tests for revo.probcal (Phase VII calibration)."""
from __future__ import annotations

import torch
import pytest

from revo.probcal import ProbCalConfig, _make_prompts


class TestProbCal:
    def test_make_prompts_length(self):
        prompts = _make_prompts(10, seed=42)
        assert len(prompts) == 10

    def test_config_defaults(self):
        cfg = ProbCalConfig()
        assert cfg.calib_frac == 0.5
        assert cfg.steps == 50
        assert cfg.lr == 0.05

    def test_seed_deterministic(self):
        p1 = _make_prompts(3, seed=0)
        p2 = _make_prompts(3, seed=0)
        assert p1 == p2

    def test_calib_frac_range(self):
        cfg = ProbCalConfig(calib_frac=0.3, alpha_curv=0.2)
        assert 0.0 < cfg.calib_frac < 1.0
        assert cfg.alpha_curv == 0.2
