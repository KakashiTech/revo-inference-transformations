"""Test HDRAM — Holographic Random Access Memory."""
import numpy as np

from revo.hdram import HDRAM, HDRAMConfig, hypertoken_hash


def test_exact_query():
    cfg = HDRAMConfig(key_dim=16, val_dim=16, max_entries=10)
    hdram = HDRAM(cfg)

    rng = np.random.RandomState(0)
    keys = [rng.randn(16).astype(np.float32) for _ in range(5)]
    vals = [rng.randn(16).astype(np.float32) for _ in range(5)]

    for k, v in zip(keys, vals):
        hdram.put(k, v)

    for k, v in zip(keys, vals):
        results = hdram.query(k, topk=1)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        retrieved, score = results[0]
        np.testing.assert_array_almost_equal(retrieved, v, decimal=5)
        assert score > 0.999, f"Expected score near 1.0, got {score}"

    print("[PASS] Exact key query returns exact value")


def test_noisy_query():
    cfg = HDRAMConfig(key_dim=16, val_dim=16, max_entries=10)
    hdram = HDRAM(cfg)

    rng = np.random.RandomState(1)
    keys = [rng.randn(16).astype(np.float32) for _ in range(5)]
    vals = [rng.randn(16).astype(np.float32) for _ in range(5)]

    for k, v in zip(keys, vals):
        hdram.put(k, v)

    for k, v in zip(keys, vals):
        noisy_k = k + 0.1 * rng.randn(16).astype(np.float32)
        results = hdram.query(noisy_k, topk=1)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        retrieved, score = results[0]
        cos_sim = float(
            np.dot(retrieved, v)
            / (np.linalg.norm(retrieved) * np.linalg.norm(v) + 1e-12)
        )
        assert cos_sim > 0.5, f"Retrieved value too far: cos_sim={cos_sim:.4f}"
        print(f"  noisy query: score={score:.4f}, value_cos={cos_sim:.4f}")

    print("[PASS] Noisy key query returns approximate value")


def test_recall():
    cfg = HDRAMConfig(key_dim=16, val_dim=16, max_entries=10)
    hdram = HDRAM(cfg)

    rng = np.random.RandomState(2)
    k = rng.randn(16).astype(np.float32)
    v = rng.randn(16).astype(np.float32)
    hdram.put(k, v)

    result = hdram.recall(k, threshold=0.95)
    assert result is not None
    np.testing.assert_array_almost_equal(result, v, decimal=5)

    noisy_k = k + 0.5 * rng.randn(16).astype(np.float32)
    result = hdram.recall(noisy_k, threshold=0.95)
    assert result is None

    print("[PASS] recall works")


def test_eviction():
    cfg = HDRAMConfig(key_dim=16, val_dim=16, max_entries=10)
    hdram = HDRAM(cfg)

    rng = np.random.RandomState(3)
    for i in range(20):
        k = rng.randn(16).astype(np.float32)
        v = rng.randn(16).astype(np.float32)
        hdram.put(k, v)
        assert hdram.size <= 10, f"size exceeded: {hdram.size}"

    assert hdram.size == 10, f"Expected size=10, got {hdram.size}"
    print(f"[PASS] Eviction keeps size <= max_entries (size={hdram.size})")


def test_hit_rate():
    cfg = HDRAMConfig(key_dim=16, val_dim=16, max_entries=10, recall_threshold=0.95)
    hdram = HDRAM(cfg)

    rng = np.random.RandomState(4)
    keys = [rng.randn(16).astype(np.float32) for _ in range(3)]
    vals = [rng.randn(16).astype(np.float32) for _ in range(3)]

    for k, v in zip(keys, vals):
        hdram.put(k, v)

    for k in keys:
        hdram.query(k)

    rate = hdram.hit_rate()
    print(f"  hit_rate={rate:.4f}")
    assert rate > 0.9, f"Expected high hit rate, got {rate}"

    print("[PASS] hit_rate works")


def test_hypertoken_hash():
    rng = np.random.RandomState(5)
    emb = rng.randn(64).astype(np.float32)

    h1 = hypertoken_hash(emb, n_bits=256)
    h2 = hypertoken_hash(emb, n_bits=256)

    assert h1 == h2, "hypertoken_hash not deterministic"
    assert isinstance(h1, str)
    assert len(h1) == 64, f"256 bits → 64 hex chars, got {len(h1)}"

    emb2 = rng.randn(64).astype(np.float32)
    h3 = hypertoken_hash(emb2, n_bits=256)
    assert h1 != h3, "Different embeddings should differ"

    print(f"[PASS] hypertoken_hash: {h1[:16]}...")


if __name__ == "__main__":
    test_exact_query()
    test_noisy_query()
    test_recall()
    test_eviction()
    test_hit_rate()
    test_hypertoken_hash()
    print("\nAll tests passed!")
