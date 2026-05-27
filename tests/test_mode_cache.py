"""Tests for revo.mode_cache."""
from __future__ import annotations

import time
from revo.mode_cache import ModeCache


class TestModeCache:
    def test_put_and_get(self):
        mc = ModeCache(ttl_seconds=3600, max_size=100)
        mc.put("key1", "value1")
        assert mc.get("key1") == "value1"

    def test_miss_returns_none(self):
        mc = ModeCache(ttl_seconds=3600, max_size=100)
        assert mc.get("nonexistent") is None

    def test_update_existing(self):
        mc = ModeCache(ttl_seconds=3600, max_size=100)
        mc.put("k", "v1")
        mc.put("k", "v2")
        assert mc.get("k") == "v2"

    def test_empty_key_ignored(self):
        mc = ModeCache(ttl_seconds=3600, max_size=100)
        mc.put("", "value")
        assert mc.get("") is None

    def test_with_signature(self):
        mc = ModeCache(ttl_seconds=3600, max_size=100)
        mc.put("k", "v", signature={"rank": 8, "ctx_dim": 64})
        v = mc.get("k")
        assert v == "v"

    def test_ttl_expiry(self):
        mc = ModeCache(ttl_seconds=0, max_size=100)  # immediate expiry
        mc.put("k", "v")
        time.sleep(0.01)
        v = mc.get("k")
        assert v is None, "TTL=0 should expire immediately"

    def test_eviction_keeps_most_recent(self):
        mc = ModeCache(ttl_seconds=3600, max_size=2)
        mc.put("a", 1)
        mc.put("b", 2)
        mc.put("c", 3)
        assert mc.get("a") is None, "oldest should be evicted"

    def test_concurrent_access(self):
        mc = ModeCache(ttl_seconds=3600, max_size=100)
        import threading
        errors = []

        def worker(i):
            try:
                mc.put(f"k{i}", f"v{i}")
                v = mc.get(f"k{i}")
                assert v == f"v{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"concurrent errors: {errors}"
