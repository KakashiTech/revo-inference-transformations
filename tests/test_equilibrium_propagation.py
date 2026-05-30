"""Tests for revo.equilibrium_propagation (EP-style tuning)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.equilibrium_propagation import equilibrium_propagation_tune
from revo.oscillatory_gating import OscillatoryHooks


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token = self
        self.pad_token = self

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        import torch
        from types import SimpleNamespace
        length = min(max_length or 8, 8)
        return SimpleNamespace(
            input_ids=torch.zeros((1, length), dtype=torch.long),
            attention_mask=torch.ones((1, length)),
        )


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.linear = nn.Linear(4, 10)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        logits = self.linear(x)
        loss = nn.functional.cross_entropy(logits.view(-1, 10), labels.view(-1), ignore_index=-100) if labels is not None else None
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(logits=logits, loss=loss)


class TestEquilibriumPropagation:
    def test_returns_dict(self):
        model = FakeModel()
        tok = FakeTokenizer()
        mgr = OscillatoryHooks()
        mgr.attach(model, name_patterns=["linear"], skip_lm_head=False)
        result = equilibrium_propagation_tune(model, tok, ["test"], mgr, steps=2, lr=1e-3)
        assert isinstance(result, dict)
        assert "steps" in result
        assert result["steps"] == 2.0

    def test_loss_delta_is_finite(self):
        model = FakeModel()
        tok = FakeTokenizer()
        mgr = OscillatoryHooks()
        mgr.attach(model, name_patterns=["linear"], skip_lm_head=False)
        result = equilibrium_propagation_tune(model, tok, ["test"], mgr, steps=2, lr=1e-3)
        assert torch.isfinite(torch.tensor(result.get("loss_delta", 0.0)))

    def test_no_phases_returns_zero_steps(self):
        model = FakeModel()
        tok = FakeTokenizer()
        mgr = OscillatoryHooks()
        # Don't attach anything
        result = equilibrium_propagation_tune(model, tok, ["test"], mgr, steps=2, lr=1e-3)
        assert result["steps"] == 0.0
