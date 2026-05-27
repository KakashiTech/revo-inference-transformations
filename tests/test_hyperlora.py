"""Tests for revo.hyperlora."""
from __future__ import annotations

import numpy as np
from revo.hyperlora import HyperLoraConfig, hyperlora_generate


class TestHyperLoraConfig:
    def test_default_config(self):
        cfg = HyperLoraConfig(context_dim=64, rank=4, in_features=128, out_features=64)
        assert cfg.context_dim == 64
        assert cfg.rank == 4
        assert cfg.in_features == 128
        assert cfg.out_features == 64

    def test_minimal_config(self):
        cfg = HyperLoraConfig(context_dim=8, rank=1, in_features=4, out_features=4)
        ctx = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        A, B, scale = hyperlora_generate(ctx, cfg)
        assert A.shape == (1, 4)
        assert B.shape == (4, 1)


class TestHyperLoraGenerate:
    def test_output_shapes(self):
        cfg = HyperLoraConfig(context_dim=64, rank=8, in_features=128, out_features=64)
        ctx = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        A, B, scale = hyperlora_generate(ctx, cfg)
        assert A.shape == (8, 128), f"A shape {A.shape} != (8, 128)"
        assert B.shape == (64, 8), f"B shape {B.shape} != (64, 8)"

    def test_deterministic(self):
        cfg = HyperLoraConfig(context_dim=32, rank=4, in_features=16, out_features=8)
        ctx = np.random.default_rng(42).standard_normal(32).astype(np.float32)
        A1, B1, s1 = hyperlora_generate(ctx, cfg)
        A2, B2, s2 = hyperlora_generate(ctx, cfg)
        assert np.allclose(A1, A2), "A not deterministic"
        assert np.allclose(B1, B2), "B not deterministic"
        assert np.isclose(s1, s2), "scale not deterministic"

    def test_diff_context_diff_output(self):
        cfg = HyperLoraConfig(context_dim=16, rank=2, in_features=8, out_features=4)
        ctx1 = np.random.default_rng(0).standard_normal(16).astype(np.float32)
        ctx2 = np.random.default_rng(1).standard_normal(16).astype(np.float32)
        A1, _, _ = hyperlora_generate(ctx1, cfg)
        A2, _, _ = hyperlora_generate(ctx2, cfg)
        assert not np.allclose(A1, A2), "different contexts should differ"

    def test_numpy_float32_compat(self):
        """Explicitly test that np.float32 input does not cause dtype errors."""
        cfg = HyperLoraConfig(context_dim=64, rank=4, in_features=6, out_features=6)
        ctx = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        A, B, scale = hyperlora_generate(ctx, cfg)
        assert A.dtype == np.float32
        assert B.dtype == np.float32

    def test_various_ranks(self):
        for rank in [1, 2, 4, 16]:
            cfg = HyperLoraConfig(context_dim=32, rank=rank, in_features=32, out_features=32)
            ctx = np.random.default_rng(0).standard_normal(32).astype(np.float32)
            A, B, scale = hyperlora_generate(ctx, cfg)
            assert A.shape == (rank, 32)
            assert B.shape == (32, rank)
            assert scale > 0

    def test_integration_with_engine(self):
        from revo.engine import apply_delta, revert_delta
        cfg = HyperLoraConfig(context_dim=64, rank=8, in_features=128, out_features=64)
        ctx = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        A, B, scale = hyperlora_generate(ctx, cfg)

        W = np.random.standard_normal((64, 128)).astype(np.float32) * 0.02
        W2, handle = apply_delta(W, A, B, scale)
        W3 = revert_delta(W2, handle)
        assert np.allclose(W, W3, atol=1e-5), "hyperlora+revert != identity"
