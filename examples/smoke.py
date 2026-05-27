"""REVO smoke test (CPU, NumPy): run `python examples/smoke.py`."""
from __future__ import annotations

import os
import numpy as np

from revo.features import build_features, build_context_vector, compute_mode_key
from revo.hyperlora import HyperLoraConfig, hyperlora_generate
from revo.engine import apply_delta, revert_delta
from revo.mode_cache import ModeCache
from revo.potentials import log_potential
from revo.observability import log_identity_anchor, log_cognitive_conservation


def main() -> None:
    prompt = "Hola REVO! Write a short list of two bullets about low-rank deltas."
    feats = build_features(prompt)
    mode_key = compute_mode_key(feats, prompt)

    ctx_dim = int(os.environ.get("REVO_CTX_DIM", "64") or 64)
    ctx = build_context_vector(feats, context_dim=ctx_dim, seed=int(os.environ.get("REVO_SEED", "0") or 0))

    in_features = 128
    out_features = 64
    cfg = HyperLoraConfig(context_dim=ctx_dim, rank=8, in_features=in_features, out_features=out_features, hidden_dim=128)
    A, B, scale = hyperlora_generate(ctx, cfg)

    np.random.seed(42)
    W = np.random.standard_normal(size=(out_features, in_features)).astype(np.float32) * 0.02

    W2, handle = apply_delta(W, A, B, scale)
    x = np.random.standard_normal(size=(in_features,)).astype(np.float32)
    y = W2 @ x
    W3 = revert_delta(W2, handle)
    assert np.allclose(W, W3, atol=1e-5), "Reversibility check failed"

    log_potential(mode_key, feats, ctx, scale)
    cache = ModeCache()
    cache.put(mode_key, ctx, signature={"rank": cfg.rank, "ctx_dim": cfg.context_dim})
    _ = cache.get(mode_key)

    output_text = "- Low-rank deltas are efficient.\n- Ephemeral application enables reversibility."
    log_identity_anchor(output_text, model="revo-demo")
    log_cognitive_conservation(output_text, model="revo-demo")

    print("OK: REVO smoke completed.")


if __name__ == "__main__":
    main()
