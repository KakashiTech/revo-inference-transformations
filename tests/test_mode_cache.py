"""Tests for ModeCache with Z-similarity search."""

from __future__ import annotations

import time
import numpy as np
import pytest

from revo.mode_cache import ModeCache


def test_exact_put_get():
    c = ModeCache(max_size=10, ttl_seconds=3600)
    c.put("k1", "v1")
    assert c.get("k1") == "v1"
    assert c.get("k2") is None


def test_similarity_hit():
    c = ModeCache(max_size=10, ttl_seconds=3600, similarity_threshold=0.9)
    z1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    z2 = np.array([0.95, 0.1, 0.05], dtype=np.float32)  # ~0.99 cos sim

    c.put("k1", "v1", z=z1)
    ctx, key, sim = c.get_similar(z2)
    assert ctx == "v1"
    assert key == "k1"
    assert sim > 0.9


def test_similarity_miss():
    c = ModeCache(max_size=10, ttl_seconds=3600, similarity_threshold=0.9)
    z1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    z2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # orthogonal = 0 sim

    c.put("k1", "v1", z=z1)
    ctx, key, sim = c.get_similar(z2)
    assert ctx is None
    assert key is None
    assert sim < 0.9


def test_similarity_chooses_best():
    c = ModeCache(max_size=10, ttl_seconds=3600, similarity_threshold=0.8)
    z_query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    c.put("far", "far_val", z=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c.put("close", "close_val", z=np.array([0.1, 0.95, 0.05], dtype=np.float32))

    ctx, key, sim = c.get_similar(z_query)
    assert ctx == "close_val"
    assert key == "close"


def test_ttl_expiry_similarity():
    c = ModeCache(max_size=10, ttl_seconds=1, similarity_threshold=0.9)
    z = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c.put("k1", "v1", z=z)
    time.sleep(1.1)
    ctx, key, sim = c.get_similar(z)
    assert ctx is None  # expired


def test_concurrent_access():
    c = ModeCache(max_size=100, ttl_seconds=3600, similarity_threshold=0.9)
    import threading
    z = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    errors = []

    def worker(i):
        try:
            c.put(f"k{i}", f"v{i}", z=z + np.random.randn(3) * 0.01)
            c.get(f"k{i}")
            c.get_similar(z)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"Concurrent access failed: {errors}"
