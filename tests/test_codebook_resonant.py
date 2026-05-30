"""Tests for revo.codebook_resonant."""
from __future__ import annotations

import numpy as np
from revo.codebook_resonant import ResonantCodebook, ResonantCodebookConfig


class TestResonantCodebookConfig:
    def test_default_config(self):
        cfg = ResonantCodebookConfig()
        assert cfg.context_dim == 16
        assert cfg.k == 64
        assert cfg.top_k == 3

    def test_custom_config(self):
        cfg = ResonantCodebookConfig(
            context_dim=32, rank=8, in_features=128, out_features=64,
            k=16, max_k=32, top_k=5,
        )
        assert cfg.context_dim == 32
        assert cfg.k == 16
        assert cfg.max_k == 32
        assert cfg.top_k == 5


class TestResonantCodebook:
    def test_init_shapes(self):
        cfg = ResonantCodebookConfig(
            context_dim=16, rank=4, in_features=32, out_features=16, k=8,
        )
        cb = ResonantCodebook(cfg)
        assert cb.codes.shape == (8, 16)
        assert cb.fitness.shape == (8,)
        assert cb.usage.shape == (8,)
        assert cb.usage.sum() == 0

    def test_query_shapes(self):
        cfg = ResonantCodebookConfig(
            context_dim=8, rank=2, in_features=4, out_features=4, k=4,
        )
        cb = ResonantCodebook(cfg)
        Z = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        Z = Z / np.linalg.norm(Z)

        A, B, scale, contrib, weights = cb.query(Z)
        assert A.shape == (2, 4)
        assert B.shape == (4, 2)
        assert isinstance(scale, float)
        assert 0.0 <= scale <= 1.0
        assert len(contrib) == min(cfg.top_k, cfg.k)
        assert len(weights) == min(cfg.top_k, cfg.k)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_query_deterministic(self):
        cfg = ResonantCodebookConfig(
            context_dim=8, rank=2, in_features=4, out_features=4, k=4,
            seed=42,
        )
        Z = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        Z = Z / np.linalg.norm(Z)

        cb1 = ResonantCodebook(cfg)
        A1, B1, s1, _, _ = cb1.query(Z)

        cb2 = ResonantCodebook(cfg)
        A2, B2, s2, _, _ = cb2.query(Z)

        assert np.allclose(A1, A2)
        assert np.allclose(B1, B2)
        assert np.isclose(s1, s2)

    def test_consolidate_improvement_moves_code(self):
        cfg = ResonantCodebookConfig(
            context_dim=4, rank=1, in_features=2, out_features=2, k=2,
            lr=0.5, top_k=1,
        )
        cb = ResonantCodebook(cfg)
        original_code = cb.codes[0].copy()

        Z = np.array([0.9, 0.3, -0.1, 0.2], dtype=np.float32)
        Z = Z / np.linalg.norm(Z)

        A, B, scale, contrib, weights = cb.query(Z)
        old_len = len(cb.codes)
        cb.consolidate(Z, A, B, scale, -0.1, contrib, weights)
        new_len = len(cb.codes)

        assert new_len >= old_len
        changed = not np.allclose(cb.codes[0], original_code, atol=1e-6)
        assert changed, "code should move toward Z on improvement"

    def test_consolidate_adds_new_code(self):
        cfg = ResonantCodebookConfig(
            context_dim=4, rank=1, in_features=2, out_features=2, k=2,
            top_k=1, add_threshold=0.99, growth_enabled=True,
            min_improvement_for_add=-0.01,
        )
        cb = ResonantCodebook(cfg)

        Z = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        A = np.zeros((1, 2), dtype=np.float32)
        B = np.zeros((2, 1), dtype=np.float32)

        old_len = len(cb.codes)
        cb.consolidate(Z, A, B, 0.5, -0.1, np.array([0], dtype=np.intp), np.array([1.0], dtype=np.float32))
        assert len(cb.codes) > old_len, "should add new code when Z is novel"

    def test_consolidate_does_not_add_on_degradation(self):
        cfg = ResonantCodebookConfig(
            context_dim=4, rank=1, in_features=2, out_features=2, k=2,
            add_threshold=0.99, growth_enabled=True,
        )
        cb = ResonantCodebook(cfg)

        Z = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        A = np.zeros((1, 2), dtype=np.float32)
        B = np.zeros((2, 1), dtype=np.float32)

        old_len = len(cb.codes)
        cb.consolidate(Z, A, B, 0.5, 0.1, np.array([0], dtype=np.intp), np.array([1.0], dtype=np.float32))
        assert len(cb.codes) == old_len, "should not add code on degradation"

    def test_growth_respects_max_k(self):
        cfg = ResonantCodebookConfig(
            context_dim=4, rank=1, in_features=2, out_features=2, k=2, max_k=3,
            add_threshold=0.0, growth_enabled=True,
            min_improvement_for_add=-1.0,
        )
        cb = ResonantCodebook(cfg)

        Z1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        Z2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        Z3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        A = np.zeros((1, 2), dtype=np.float32)
        B = np.zeros((2, 1), dtype=np.float32)

        for Z in [Z1, Z2, Z3]:
            _, _, _, contrib, weights = cb.query(Z)
            cb.consolidate(Z, A, B, 0.5, -0.2, contrib, weights)

        assert len(cb.codes) <= cfg.max_k

    def test_state_dict_roundtrip(self):
        cfg = ResonantCodebookConfig(
            context_dim=4, rank=1, in_features=2, out_features=2, k=2,
        )
        cb1 = ResonantCodebook(cfg)
        Z = np.random.default_rng(0).standard_normal(4).astype(np.float32)
        Z = Z / np.linalg.norm(Z)
        A, B, scale, contrib, weights = cb1.query(Z)
        cb1.consolidate(Z, A, B, scale, -0.05, contrib, weights)

        sd = cb1.state_dict()

        cb2 = ResonantCodebook(cfg)
        cb2.load_state_dict(sd)

        assert np.allclose(cb1.codes, cb2.codes)
        assert np.allclose(cb1.fitness, cb2.fitness)
        assert np.allclose(cb1.usage, cb2.usage)
        assert cb1.stats()["codebook_size"] == cb2.stats()["codebook_size"]

    def test_stats_structure(self):
        cfg = ResonantCodebookConfig(context_dim=4, rank=1, in_features=2, out_features=2, k=4)
        cb = ResonantCodebook(cfg)
        s = cb.stats()
        assert "codebook_size" in s
        assert "total_usage" in s
        assert "mean_fitness" in s
        assert s["codebook_size"] == 4

    def test_prune_removes_bad_codes(self):
        cfg = ResonantCodebookConfig(
            context_dim=4, rank=1, in_features=2, out_features=2, k=8,
            prune_interval=1, prune_min_usage=0, prune_max_fitness=-0.01,
        )
        cb = ResonantCodebook(cfg)
        cb.fitness[:] = 0.05
        cb.usage[:] = 0
        cb._step_counter = 1
        cb._prune()
        assert len(cb.codes) >= 4

    def test_query_increases_usage(self):
        cfg = ResonantCodebookConfig(
            context_dim=4, rank=1, in_features=2, out_features=2, k=4,
        )
        cb = ResonantCodebook(cfg)
        Z = np.random.default_rng(0).standard_normal(4).astype(np.float32)
        Z = Z / np.linalg.norm(Z)

        before = cb.usage.sum()
        cb.query(Z)
        after = cb.usage.sum()
        assert after > before

    def test_query_includes_mutation_on_novelty(self):
        cfg = ResonantCodebookConfig(
            context_dim=4, rank=1, in_features=2, out_features=2, k=1,
            exploration_scale=1.0,
        )
        cb = ResonantCodebook(cfg)

        Z_novel = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        A1, B1, s1, _, _ = cb.query(Z_novel)
        A2, B2, s2, _, _ = cb.query(Z_novel)

        assert not np.allclose(A1, A2), "high novelty should add mutation noise"
