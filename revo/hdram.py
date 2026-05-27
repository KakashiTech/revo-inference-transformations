"""HDRAM — Holographic Random Access Memory.

Content-addressable associative memory using NumPy cosine similarity.
Replaces the KV-cache with learned key-value retrieval (approximate nearest
neighbor search with physics-inspired naming).

ADR: The "holographic" naming reflects that each stored vector is distributed
across the entire key space — retrieval is a content-addressable lookup via
cosine similarity, analogous to reading a hologram. This is *not* true optical
holography; it is ANN search with a poetic name.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HDRAMConfig:
    key_dim: int = 64
    val_dim: int = 64
    max_entries: int = 1024
    recall_threshold: float = 0.95


_RNG_SEED = 42


class HDRAM:
    """Holographic Random Access Memory — associative key-value store.

    Stores (key, value) vector pairs in pre-allocated NumPy arrays and
    retrieves values by cosine similarity between the probe and stored keys.
    """

    def __init__(self, cfg: HDRAMConfig):
        self.cfg = cfg
        self.keys = np.zeros((cfg.max_entries, cfg.key_dim), dtype=np.float32)
        self.values = np.zeros((cfg.max_entries, cfg.val_dim), dtype=np.float32)
        self._age = np.zeros(cfg.max_entries, dtype=np.int64)
        self.size = 0
        self._clock = 0
        self._query_log: deque = deque(maxlen=1000)

    # ------------------------------------------------------------------
    #  internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.reshape(1, -1) if a.ndim == 1 else a
        a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
        return (a @ b.T) / (a_norm @ b_norm.T)

    def _find_slot(self, key: np.ndarray) -> Tuple[Optional[int], float]:
        if self.size == 0:
            return None, 0.0
        sims = self._cosine_sim(key, self.keys[:self.size])
        idx = int(np.argmax(sims[0]))
        return idx, float(sims[0, idx])

    # ------------------------------------------------------------------
    #  public API
    # ------------------------------------------------------------------

    def put(self, key: np.ndarray, value: np.ndarray) -> None:
        key = np.ascontiguousarray(key, dtype=np.float32).ravel()
        value = np.ascontiguousarray(value, dtype=np.float32).ravel()
        assert key.shape[0] == self.cfg.key_dim
        assert value.shape[0] == self.cfg.val_dim

        idx, score = self._find_slot(key)
        if idx is not None and score > 0.999:
            self.values[idx] = value
            self._age[idx] = self._clock
            self._clock += 1
            return

        if self.size >= self.cfg.max_entries:
            self.evict()

        slot = self.size
        self.keys[slot] = key
        self.values[slot] = value
        self._age[slot] = self._clock
        self._clock += 1
        self.size += 1

    def query(self, probe: np.ndarray, topk: int = 1) -> List[Tuple[np.ndarray, float]]:
        probe = np.ascontiguousarray(probe, dtype=np.float32).ravel()
        if self.size == 0:
            return []

        sims = self._cosine_sim(probe, self.keys[:self.size])
        indices = np.argsort(-sims[0])[:topk]
        results = [(self.values[i].copy(), float(sims[0, i])) for i in indices]

        hit = 1 if results and results[0][1] >= self.cfg.recall_threshold else 0
        self._query_log.append(hit)
        return results

    def recall(self, probe: np.ndarray,
               threshold: Optional[float] = None) -> Optional[np.ndarray]:
        probe = np.ascontiguousarray(probe, dtype=np.float32).ravel()
        if self.size == 0:
            return None
        th = threshold if threshold is not None else self.cfg.recall_threshold
        sims = self._cosine_sim(probe, self.keys[:self.size])
        idx = int(np.argmax(sims[0]))
        if sims[0, idx] >= th:
            return self.values[idx].copy()
        return None

    def evict(self, policy: str = 'lru') -> None:
        if self.size == 0:
            return
        if policy == 'lru':
            oldest = int(np.argmin(self._age[:self.size]))
            if oldest < self.size - 1:
                n = self.size - oldest - 1
                self.keys[oldest:oldest + n] = self.keys[oldest + 1:self.size]
                self.values[oldest:oldest + n] = self.values[oldest + 1:self.size]
                self._age[oldest:oldest + n] = self._age[oldest + 1:self.size]
            self.size -= 1
            self.keys[self.size] = 0
            self.values[self.size] = 0
            self._age[self.size] = 0
        else:
            raise ValueError(f"Unknown eviction policy: {policy!r}")

    def hit_rate(self, window: int = 100) -> float:
        if not self._query_log:
            return 0.0
        recent = list(self._query_log)[-window:]
        return sum(recent) / len(recent)


def hypertoken_hash(embedding: np.ndarray, n_bits: int = 256) -> str:
    """Simhash: project embedding onto random binary hyperplanes → hex string.

    Deterministic given a fixed internal seed (42).
    """
    emb = np.ascontiguousarray(embedding, dtype=np.float32).ravel()
    rng = np.random.RandomState(_RNG_SEED)
    hyperplanes = rng.randn(n_bits, emb.shape[0]).astype(np.float32)
    bits = ((hyperplanes @ emb) >= 0).astype(np.uint8)

    n_bytes = (n_bits + 7) // 8
    padded = np.zeros(n_bytes * 8, dtype=np.uint8)
    padded[:n_bits] = bits
    return np.packbits(padded).tobytes().hex()[: (n_bits + 3) // 4]
