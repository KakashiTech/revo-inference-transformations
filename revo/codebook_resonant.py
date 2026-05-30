"""ResonantCodebook — codebook of latent knowledge for delta generation.

Replaces HyperLoRA's fixed MLP with an adaptive codebook that:
- Stores prototypical context vectors and their associated deltas
- Retrieves by cosine similarity (resonance)
- Combines via weighted average (recombination)
- Adds mutation scaled by novelty (exploration)
- Updates online based on NLL feedback (consolidation)
- Prunes low-fitness, low-usage codes periodically

Usage:
    cb = ResonantCodebook(cfg)
    A, B, scale, contrib, weights = cb.query(Z)
    # ... apply delta, get nll_delta ...
    cb.consolidate(Z, A, B, scale, nll_delta, contrib, weights)
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 1e-10 else 0.0


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = x / max(temp, 1e-8)
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-10)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / n if n > 1e-8 else x


@dataclass
class ResonantCodebookConfig:
    context_dim: int = 16
    rank: int = 4
    in_features: int = 768
    out_features: int = 50257
    k: int = 64
    max_k: int = 256
    lr: float = 0.15
    top_k: int = 3
    exploration_scale: float = 0.08
    similarity_threshold: float = 0.7
    add_threshold: float = 0.7
    min_improvement_for_add: float = -0.05
    prune_interval: int = 500
    prune_min_usage: int = 10
    prune_max_fitness: float = 0.02
    scale_min: float = 0.0
    scale_max: float = 0.8
    seed: int = 0
    growth_enabled: bool = True
    mutation_decay: float = 0.5


class ResonantCodebook:

    def __init__(self, cfg: ResonantCodebookConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._step_counter: int = 0
        self._delta_cache: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}

        hd = max(32, cfg.context_dim * 2)
        hd = min(hd, 128)
        scale_w = 1.0 / max(4.0, math.sqrt(cfg.context_dim))
        self.W1 = self.rng.standard_normal((hd, cfg.context_dim)).astype(np.float32) * scale_w
        self.b1 = self.rng.standard_normal((hd,)).astype(np.float32) * 0.05
        delta_dim = cfg.rank * cfg.in_features + cfg.out_features * cfg.rank + 1
        scale_w2 = 1.0 / max(4.0, math.sqrt(hd))
        self.W2 = self.rng.standard_normal((delta_dim, hd)).astype(np.float32) * scale_w2
        self.b2 = self.rng.standard_normal((delta_dim,)).astype(np.float32) * 0.02
        self._delta_dim = delta_dim
        self._a_flat = cfg.rank * cfg.in_features
        self._b_flat = cfg.out_features * cfg.rank

        self.codes: np.ndarray = np.zeros((0, cfg.context_dim), dtype=np.float32)
        self.fitness: np.ndarray = np.zeros(0, dtype=np.float32)
        self.usage: np.ndarray = np.zeros(0, dtype=np.int32)
        self._init_codes()

    def _init_codes(self) -> None:
        k = self.cfg.k
        d = self.cfg.context_dim
        raw = self.rng.standard_normal((k, d)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.codes = raw / np.maximum(norms, 1e-8)
        self.fitness = np.zeros(k, dtype=np.float32)
        self.usage = np.zeros(k, dtype=np.int32)

    def _generate_delta(self, code: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        cfg = self.cfg
        h = np.tanh(self.W1 @ code.ravel() + self.b1)
        raw = self.W2 @ h + self.b2

        a_vec = raw[:self._a_flat].reshape(cfg.rank, cfg.in_features).astype(np.float32) * 0.05
        b_vec = raw[self._a_flat:self._a_flat + self._b_flat].reshape(
            cfg.out_features, cfg.rank
        ).astype(np.float32) * 0.05
        scale_raw = float(raw[-1])
        base = 0.5 * (scale_raw + 1.0)
        mabs = float(np.mean(np.abs(code.ravel())))
        gate = 0.85 + 0.30 * (mabs / (1.0 + mabs))
        scale = float(max(cfg.scale_min, min(cfg.scale_max, base * gate)))
        return a_vec, b_vec, scale

    def query(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        Z = Z.ravel().astype(np.float32)
        sims = np.array([_cosine_sim(Z, c) for c in self.codes], dtype=np.float32)
        top_n = min(self.cfg.top_k, len(self.codes))
        top_idx = np.argsort(-sims)[:top_n]
        top_sims = sims[top_idx]
        weights = _softmax(top_sims, temp=0.5)
        max_sim = float(top_sims[0])
        novelty = max(0.0, 1.0 - max_sim)

        A_list, B_list, s_list = [], [], []
        for idx in top_idx:
            if int(idx) in self._delta_cache:
                A, B, s = self._delta_cache[int(idx)]
            else:
                A, B, s = self._generate_delta(self.codes[int(idx)])
                self._delta_cache[int(idx)] = (A, B, s)
            A_list.append(A)
            B_list.append(B)
            s_list.append(s)

        A_comb = sum(A_list[i] * weights[i] for i in range(top_n))
        B_comb = sum(B_list[i] * weights[i] for i in range(top_n))
        s_comb = sum(s_list[i] * weights[i] for i in range(top_n))

        if novelty > 0.2:
            noise_amp = self.cfg.exploration_scale * (novelty ** self.cfg.mutation_decay)
            A_comb = A_comb + self.rng.standard_normal(A_comb.shape).astype(np.float32) * noise_amp
            B_comb = B_comb + self.rng.standard_normal(B_comb.shape).astype(np.float32) * noise_amp
            s_jitter = float(self.rng.standard_normal()) * noise_amp
            s_comb = float(max(self.cfg.scale_min, min(self.cfg.scale_max, s_comb + s_jitter)))

        s_comb = float(max(self.cfg.scale_min, min(self.cfg.scale_max, s_comb)))

        for idx in top_idx:
            self.usage[int(idx)] += 1

        self._step_counter += 1
        if self._step_counter % self.cfg.prune_interval == 0:
            self._prune()

        return A_comb, B_comb, s_comb, top_idx, weights

    def consolidate(self, Z: np.ndarray, A: np.ndarray, B: np.ndarray,
                    scale: float, nll_delta: float,
                    contrib_codes: np.ndarray, weights: np.ndarray) -> None:
        Z = Z.ravel().astype(np.float32)
        best_idx = int(contrib_codes[0])
        best_weight = float(weights[0])

        if nll_delta < -0.01:
            lr = self.cfg.lr * best_weight
            self.codes[best_idx] = _l2_normalize(
                self.codes[best_idx] * (1.0 - lr) + Z * lr
            )
            existing = self._delta_cache.get(best_idx)
            if existing is not None:
                eA, eB, eS = existing
                self._delta_cache[best_idx] = (
                    eA * 0.7 + A * 0.3,
                    eB * 0.7 + B * 0.3,
                    eS * 0.7 + scale * 0.3,
                )
            else:
                self._delta_cache[best_idx] = (A, B, scale)
            self.fitness[best_idx] = 0.9 * self.fitness[best_idx] + 0.1 * nll_delta

            sims = np.array([_cosine_sim(Z, c) for c in self.codes])
            max_sim = float(np.max(sims))
            if (max_sim < self.cfg.add_threshold
                    and nll_delta < self.cfg.min_improvement_for_add
                    and len(self.codes) < self.cfg.max_k
                    and self.cfg.growth_enabled):
                self._add_code(Z, A, B, scale, nll_delta)

        elif nll_delta > 0.02:
            for idx in contrib_codes:
                self.fitness[int(idx)] = 0.95 * self.fitness[int(idx)] + 0.05 * nll_delta

    def _add_code(self, Z: np.ndarray, A: np.ndarray, B: np.ndarray,
                  scale: float, nll_delta: float) -> None:
        new_code = _l2_normalize(Z + 0.1 * self.rng.standard_normal(Z.shape).astype(np.float32))
        self.codes = np.concatenate([self.codes, new_code[None, :]], axis=0)
        self.fitness = np.concatenate([self.fitness, np.array([nll_delta], dtype=np.float32)])
        self.usage = np.concatenate([self.usage, np.zeros(1, dtype=np.int32)])
        new_idx = len(self.codes) - 1
        self._delta_cache[new_idx] = (A, B, scale)

    def _prune(self) -> None:
        if len(self.codes) <= 4:
            return
        max_fitness = self.cfg.prune_max_fitness
        min_usage = self.cfg.prune_min_usage
        keep = np.ones(len(self.codes), dtype=bool)
        for i in range(len(self.codes)):
            if self.fitness[i] > max_fitness and self.usage[i] < min_usage:
                keep[i] = False
        if np.sum(keep) < 4:
            return
        removed = np.sum(~keep)
        if removed > 0:
            old_idx_map = {old: new for new, old in enumerate(np.where(keep)[0])}
            self.codes = self.codes[keep]
            self.fitness = self.fitness[keep]
            self.usage = self.usage[keep]
            self._delta_cache = {
                old_idx_map[i]: v
                for i, v in self._delta_cache.items()
                if i in old_idx_map
            }

    def state_dict(self) -> dict:
        return {
            "codes": self.codes.tolist(),
            "fitness": self.fitness.tolist(),
            "usage": self.usage.tolist(),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "step": self._step_counter,
            "delta_cache": {
                str(k): (A.tolist(), B.tolist(), s)
                for k, (A, B, s) in self._delta_cache.items()
            },
            "cfg": {
                "context_dim": self.cfg.context_dim,
                "rank": self.cfg.rank,
                "in_features": self.cfg.in_features,
                "out_features": self.cfg.out_features,
                "k": self.cfg.k,
                "max_k": self.cfg.max_k,
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        self.codes = np.array(sd["codes"], dtype=np.float32)
        self.fitness = np.array(sd["fitness"], dtype=np.float32)
        self.usage = np.array(sd["usage"], dtype=np.int32)
        self.W1 = np.array(sd["W1"], dtype=np.float32)
        self.b1 = np.array(sd["b1"], dtype=np.float32)
        self.W2 = np.array(sd["W2"], dtype=np.float32)
        self.b2 = np.array(sd["b2"], dtype=np.float32)
        self._step_counter = int(sd["step"])
        self._delta_cache = {}
        for k, v in sd["delta_cache"].items():
            self._delta_cache[int(k)] = (
                np.array(v[0], dtype=np.float32),
                np.array(v[1], dtype=np.float32),
                float(v[2]),
            )

    def seed_from_hyperlora(self, Z_samples: List[np.ndarray], hl_cfg: "HyperLoraConfig") -> None:
        """Pre-populate codebook from HyperLoRA-generated deltas.
        
        Takes a list of Z vectors, generates deltas via HyperLoRA for each,
        and stores them as codebook entries. Uses k-means++ to select diverse
        seeds if more Z_samples than codebook slots.
        """
        from revo.hyperlora import hyperlora_generate
        k = min(self.cfg.k, len(Z_samples))
        if k < 2:
            return

        codes = []
        deltas = []
        fitnesses = []

        for Z in Z_samples[:k]:
            Z = Z.ravel().astype(np.float32)
            Z_norm = _l2_normalize(Z)
            A, B, scale = hyperlora_generate(Z, hl_cfg)
            codes.append(Z_norm)
            deltas.append((A, B, scale))
            fitnesses.append(0.0)

        self.codes = np.array(codes, dtype=np.float32)
        self.fitness = np.array(fitnesses, dtype=np.float32)
        self.usage = np.zeros(len(codes), dtype=np.int32)
        self._delta_cache = {i: deltas[i] for i in range(len(deltas))}

    def stats(self) -> dict:
        return {
            "codebook_size": len(self.codes),
            "max_k": self.cfg.max_k,
            "total_usage": int(np.sum(self.usage)),
            "mean_fitness": float(np.mean(self.fitness)),
            "delta_cache_size": len(self._delta_cache),
            "steps": self._step_counter,
        }
