"""Tests for revo.features."""
from __future__ import annotations

from revo.features import build_features, compute_mode_key, build_context_vector


class TestBuildFeatures:
    def test_basic_structure(self):
        feats = build_features("hello world")
        assert isinstance(feats, dict)
        assert len(feats) > 0

    def test_empty_prompt_doesnt_crash(self):
        feats = build_features("")
        assert isinstance(feats, dict)

    def test_keys_are_strings(self):
        feats = build_features("test prompt")
        assert all(isinstance(k, str) for k in feats)


class TestComputeModeKey:
    def test_deterministic(self):
        k1 = compute_mode_key({"a": 1}, "hello")
        k2 = compute_mode_key({"a": 1}, "hello")
        assert k1 == k2, "not deterministic"

    def test_different_prompts_differ(self):
        k1 = compute_mode_key({"a": 1}, "hello")
        k2 = compute_mode_key({"a": 1}, "world")
        assert k1 != k2, "different prompts should differ"

    def test_different_feats_differ(self):
        k1 = compute_mode_key({"a": 1}, "hello")
        k2 = compute_mode_key({"b": 2}, "hello")
        assert k1 != k2, "different feats should differ"

    def test_unicode_normalization(self):
        """NFC and NFD forms of the same text must produce identical keys."""
        k1 = compute_mode_key({"a": 1}, "café")           # NFC
        k2 = compute_mode_key({"a": 1}, "cafe\u0301")     # NFD
        assert k1 == k2, "NFC/NFD mismatch"

    def test_length_param(self):
        k1 = compute_mode_key({"a": 1}, "hello", take=8)
        k2 = compute_mode_key({"a": 1}, "hello", take=16)
        assert len(k1) == 8
        assert len(k2) == 16

    def test_head_truncation(self):
        long = "x" * 500
        k = compute_mode_key({"a": 1}, long)
        assert len(k) == 24  # default take


class TestBuildContextVector:
    def test_output_shape(self):
        feats = build_features("test")
        ctx = build_context_vector(feats, context_dim=64, seed=0)
        assert ctx.shape == (64,)

    def test_deterministic(self):
        feats = build_features("test")
        c1 = build_context_vector(feats, context_dim=32, seed=42)
        c2 = build_context_vector(feats, context_dim=32, seed=42)
        import numpy as np
        assert np.allclose(c1, c2)

    def test_diff_seeds_differ(self):
        feats = build_features("test")
        import numpy as np
        c1 = build_context_vector(feats, context_dim=32, seed=0)
        c2 = build_context_vector(feats, context_dim=32, seed=1)
        assert not np.allclose(c1, c2)
