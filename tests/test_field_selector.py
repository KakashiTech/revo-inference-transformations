"""Tests for revo.field_selector."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from revo.field_selector import FieldSelector, FieldSelectorConfig, discover_modules


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(64, 100)
        self.transformer = nn.ModuleDict({
            "h": nn.ModuleList([
                nn.ModuleDict({
                    "attn": nn.ModuleDict({
                        "c_attn": nn.Linear(64, 192),
                        "c_proj": nn.Linear(64, 64),
                    }),
                    "mlp": nn.ModuleDict({
                        "c_fc": nn.Linear(64, 256),
                        "c_proj": nn.Linear(256, 64),
                    }),
                })
                for _ in range(3)
            ]),
            "wte": nn.Embedding(100, 64),
            "wpe": nn.Embedding(100, 64),
            "ln_f": nn.LayerNorm(64),
        })


class TestDiscoverModules:
    def test_discover_lm_head(self):
        model = DummyModel()
        mods = discover_modules(
            model, include_patterns=("h.", "lm_head"), exclude_patterns=("wte", "wpe", "ln_")
        )
        names = [m[0] for m in mods]
        assert any("lm_head" in n for n in names)
        assert any("h.0.attn.c_attn" in n for n in names)

    def test_count_modules(self):
        model = DummyModel()
        mods = discover_modules(
            model, include_patterns=("h.", "lm_head"), exclude_patterns=("wte", "wpe", "ln_")
        )
        assert len(mods) == 1 + 3 * 4  # lm_head + 3 layers * 4 modules each


class TestFieldSelector:
    def test_init(self):
        model = DummyModel()
        cfg = FieldSelectorConfig(context_dim=8, embed_dim=4, hidden_dim=16)
        fs = FieldSelector(model, cfg)
        assert fs.n_modules > 0

    def test_select_returns_list(self):
        model = DummyModel()
        cfg = FieldSelectorConfig(
            context_dim=8, embed_dim=4, hidden_dim=16,
            max_active_modules=4, score_threshold=-0.5,
        )
        fs = FieldSelector(model, cfg)
        Z = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        Z = Z / np.linalg.norm(Z)
        selected = fs.select(Z)
        assert isinstance(selected, list)
        assert len(selected) <= 4
        if selected:
            idx, name, score, intensity = selected[0]
            assert isinstance(idx, int)
            assert isinstance(name, str)
            assert isinstance(score, float)
            assert 0.0 <= intensity <= 1.0

    def test_get_active_distribution(self):
        model = DummyModel()
        cfg = FieldSelectorConfig(context_dim=8, embed_dim=4, hidden_dim=16)
        fs = FieldSelector(model, cfg)
        Z = np.random.default_rng(42).standard_normal(8).astype(np.float32)
        Z = Z / np.linalg.norm(Z)
        dist = fs.get_active_distribution(Z)
        assert "n_selected" in dist
        assert "layer_distribution" in dist

    def test_deterministic_same_Z(self):
        model = DummyModel()
        cfg = FieldSelectorConfig(
            context_dim=8, embed_dim=4, hidden_dim=16, seed=0,
        )
        Z = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        Z = Z / np.linalg.norm(Z)

        fs1 = FieldSelector(model, cfg)
        s1 = fs1.select(Z)

        fs2 = FieldSelector(model, cfg)
        s2 = fs2.select(Z)

        assert len(s1) == len(s2)
        for (i1, n1, sc1, in1), (i2, n2, sc2, in2) in zip(s1, s2):
            assert i1 == i2
            assert n1 == n2

    def test_select_different_for_different_Z(self):
        model = DummyModel()
        cfg = FieldSelectorConfig(context_dim=8, embed_dim=4, hidden_dim=16, max_active_modules=3)
        fs = FieldSelector(model, cfg)

        Z1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        Z2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        s1 = fs.select(Z1)
        s2 = fs.select(Z2)

        names1 = {n for _, n, _, _ in s1}
        names2 = {n for _, n, _, _ in s2}
        assert names1 != names2, "different Z should select different modules"

    def test_stats_structure(self):
        model = DummyModel()
        cfg = FieldSelectorConfig(context_dim=8, embed_dim=4, hidden_dim=16)
        fs = FieldSelector(model, cfg)
        s = fs.stats()
        assert "n_modules" in s
        assert "module_names" in s
        assert len(s["module_names"]) == fs.n_modules
