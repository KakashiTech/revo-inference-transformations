from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class HyperLoraConfig:
    context_dim: int
    rank: int
    in_features: int
    out_features: int
    hidden_dim: int = 128
    scale_min: float = 0.5
    scale_max: float = 1.2
    seed: int = 0


def _stable_seed(ctx: np.ndarray, cfg: HyperLoraConfig) -> int:
    base_seed = int(os.environ.get("REVO_SEED", "0") or 0) ^ int(cfg.seed or 0)
    try:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(ctx.astype(np.float32, copy=False)).tobytes())
        h.update(int(cfg.context_dim).to_bytes(4, "little", signed=False))
        h.update(int(cfg.rank).to_bytes(4, "little", signed=False))
        h.update(int(cfg.in_features).to_bytes(4, "little", signed=False))
        h.update(int(cfg.out_features).to_bytes(4, "little", signed=False))
        h.update(int(cfg.hidden_dim).to_bytes(4, "little", signed=False))
        digest = h.digest()
        mix = int.from_bytes(digest[:8], "little") ^ int.from_bytes(digest[8:16], "little")
        return (base_seed ^ mix) & 0x7FFFFFFF
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("_stable_seed ctx hash failed: %s, falling back to base_seed (non-deterministic)", e)
        return base_seed & 0x7FFFFFFF


def _mlp(ctx: np.ndarray, rng: np.random.Generator, cfg: HyperLoraConfig, out_dim: int) -> np.ndarray:
    # Two-layer tanh MLP; weights sampled deterministically from rng
    W1 = rng.standard_normal(size=(cfg.hidden_dim, cfg.context_dim)).astype(np.float32) * (1.0 / max(4.0, np.sqrt(cfg.context_dim)))
    b1 = rng.standard_normal(size=(cfg.hidden_dim,)).astype(np.float32) * 0.05
    h1 = np.tanh(W1 @ ctx.reshape(-1).astype(np.float32) + b1)
    W2 = rng.standard_normal(size=(out_dim, cfg.hidden_dim)).astype(np.float32) * (1.0 / max(4.0, np.sqrt(cfg.hidden_dim)))
    b2 = rng.standard_normal(size=(out_dim,)).astype(np.float32) * 0.02
    z = W2 @ h1 + b2
    return np.tanh(z).astype(np.float32)


def hyperlora_generate(ctx: np.ndarray, cfg: HyperLoraConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """Map context -> (A, B, scale) for an ephemeral low-rank delta.

    Shapes: A=(rank,in_features), B=(out_features,rank); delta=(B@A)*scale
    """
    assert int(ctx.size) == int(cfg.context_dim), "ctx size must match context_dim"
    seed = _stable_seed(ctx, cfg)
    rng = np.random.default_rng(seed)

    a_flat = cfg.rank * cfg.in_features
    b_flat = cfg.out_features * cfg.rank
    out_dim = a_flat + b_flat + 1

    z = _mlp(ctx, rng, cfg, out_dim)

    a_vec = z[:a_flat]
    b_vec = z[a_flat:a_flat + b_flat]
    scale_raw = float(z[-1])

    a_vec = a_vec * 0.05
    b_vec = b_vec * 0.05

    A = a_vec.reshape(cfg.rank, cfg.in_features).astype(np.float32)
    B = b_vec.reshape(cfg.out_features, cfg.rank).astype(np.float32)

    base = 0.5 * (scale_raw + 1.0)
    mabs = float(np.mean(np.abs(ctx.astype(np.float32))))
    gate = 0.85 + 0.30 * (mabs / (1.0 + mabs))

    scale = float(base * gate)
    scale = float(max(cfg.scale_min, min(cfg.scale_max, scale)))

    return A, B, scale
