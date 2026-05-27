"""Tests for revo.potentials."""
from __future__ import annotations

import os
import json
import numpy as np
from revo.potentials import log_potential, _q16_b64


class TestQ16B64:
    def test_basic_encoding(self):
        ctx = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        encoded = _q16_b64(ctx)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

    def test_ascii_only(self):
        ctx = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        encoded = _q16_b64(ctx)
        assert all(ord(c) < 128 for c in encoded), "non-ASCII in b64"

    def test_empty_array(self):
        ctx = np.array([], dtype=np.float32)
        encoded = _q16_b64(ctx)

    def test_float16_fallback(self):
        ctx = np.array([float('nan'), float('inf')], dtype=np.float32)
        encoded = _q16_b64(ctx)
        assert isinstance(encoded, str)


class TestLogPotential:
    def test_writes_jsonl(self, tmp_path):
        log_path = os.path.join(tmp_path, "potentials.jsonl")
        ctx = np.random.default_rng(0).standard_normal(64).astype(np.float32)
        log_potential("mode123", {"feat_a": 1.0}, ctx, 0.5, log_path=str(log_path))
        assert os.path.exists(log_path)
        with open(log_path) as f:
            line = json.loads(f.readline())
        assert line["mode_key"] == "mode123"
        assert line["scale"] == 0.5

    def test_codebook_write(self, tmp_path):
        log_path = os.path.join(tmp_path, "pot.jsonl")
        cb_path = os.path.join(tmp_path, "codebook.jsonl")
        ctx = np.random.default_rng(0).standard_normal(16).astype(np.float32)
        log_potential("mk", {"a": 1}, ctx, 1.0, log_path=str(log_path), codebook_path=str(cb_path))
        assert os.path.exists(cb_path)

    def test_skip_codebook(self, tmp_path):
        log_path = os.path.join(tmp_path, "p.jsonl")
        ctx = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        log_potential("mk", {}, ctx, 0.1, log_path=str(log_path), include_codebook=False)
        assert os.path.exists(log_path)

    def test_rotation(self, tmp_path):
        log_path = os.path.join(tmp_path, "big.jsonl")
        with open(log_path, "w") as f:
            f.write("x" * 100)
        assert os.path.getsize(log_path) == 100
        ctx = np.random.default_rng(0).standard_normal(8).astype(np.float32)
        log_potential("k", {}, ctx, 1.0, log_path=str(log_path))
        assert os.path.exists(log_path)
