"""Tests for revo.biocomp (Phase IX evaluation)."""
from __future__ import annotations

import torch
import pytest

from revo.biocomp import BioCompConfig, _make_prompts, _hidden_metrics


class TestBioComp:
    def test_make_prompts_length(self):
        prompts = _make_prompts(5, seed=42)
        assert len(prompts) == 5

    def test_seed_deterministic(self):
        p1 = _make_prompts(3, seed=0)
        p2 = _make_prompts(3, seed=0)
        assert p1 == p2

    def test_config_defaults(self):
        cfg = BioCompConfig()
        assert cfg.act_quantile == 0.75
        assert cfg.topk_eval == 64

    @pytest.mark.slow
    def test_hidden_metrics_returns_floats(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model.eval()
        act, coh = _hidden_metrics(model, tok, ["test prompt"], max_length=16, act_q=0.75)
        assert isinstance(act, float)
        assert isinstance(coh, float)
        assert 0.0 <= act <= 1.0
        assert 0.0 <= coh <= 1.0
