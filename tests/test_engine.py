"""Tests for revo.engine: apply_delta / revert_delta."""
from __future__ import annotations

import numpy as np
from revo.engine import apply_delta, revert_delta


class TestApplyRevert:
    def test_roundtrip_identity(self, rng):
        W = rng.standard_normal((16, 32)).astype(np.float32) * 0.02
        A = rng.standard_normal((4, 32)).astype(np.float32) * 0.01
        B = rng.standard_normal((16, 4)).astype(np.float32) * 0.01
        scale = 0.5

        W2, handle = apply_delta(W, A, B, scale)
        W3 = revert_delta(W2, handle)

        assert np.allclose(W, W3, atol=1e-5), "apply+revert != identity"

    def test_multiple_applies_and_reverts(self, rng):
        W = rng.standard_normal((8, 8)).astype(np.float32)
        A1 = rng.standard_normal((2, 8)).astype(np.float32)
        B1 = rng.standard_normal((8, 2)).astype(np.float32)
        A2 = rng.standard_normal((3, 8)).astype(np.float32)
        B2 = rng.standard_normal((8, 3)).astype(np.float32)

        W1, h1 = apply_delta(W, A1, B1, 1.0)
        W2, h2 = apply_delta(W1, A2, B2, 1.0)
        W3 = revert_delta(W2, h1)
        W4 = revert_delta(W3, h2)

        assert np.allclose(W, W4, atol=1e-5), "multi-layer revert != identity"

    def test_zero_scale_no_change(self, rng):
        W = rng.standard_normal((4, 4)).astype(np.float32)
        A = rng.standard_normal((2, 4)).astype(np.float32)
        B = rng.standard_normal((4, 2)).astype(np.float32)

        W2, handle = apply_delta(W, A, B, 0.0)
        assert np.allclose(W, W2, atol=1e-6), "scale=0 should not change W"

    def test_different_shapes(self, rng):
        W = rng.standard_normal((64, 128)).astype(np.float32)
        A = rng.standard_normal((8, 128)).astype(np.float32)
        B = rng.standard_normal((64, 8)).astype(np.float32)

        W2, handle = apply_delta(W, A, B, 0.1)
        W3 = revert_delta(W2, handle)
        assert np.allclose(W, W3, atol=1e-5), f"shape ({W.shape}) roundtrip failed"

    def test_delta_handle_structure(self, rng):
        W = rng.standard_normal((4, 4)).astype(np.float32)
        A = rng.standard_normal((1, 4)).astype(np.float32)
        B = rng.standard_normal((4, 1)).astype(np.float32)

        _, handle = apply_delta(W, A, B, 1.0)
        assert hasattr(handle, 'delta'), "handle missing delta"
        assert hasattr(handle, 'scale'), "handle missing scale"
        assert handle.delta.shape == (4, 4)
