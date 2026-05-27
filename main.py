from __future__ import annotations

import os
import argparse
import numpy as np

from revo.features import build_features, build_context_vector, compute_mode_key
from revo.hyperlora import HyperLoraConfig, hyperlora_generate
from revo.engine import apply_delta, revert_delta
from revo.mode_cache import ModeCache
from revo.potentials import log_potential
from revo.observability import log_identity_anchor, log_cognitive_conservation


def run(prompt: str) -> None:
    feats = build_features(prompt)
    mode_key = compute_mode_key(feats, prompt)
    ctx_dim = int(os.environ.get("REVO_CTX_DIM", "64") or 64)
    seed = int(os.environ.get("REVO_SEED", "0") or 0)
    ctx = build_context_vector(feats, context_dim=ctx_dim, seed=seed)
    in_features = int(os.environ.get("REVO_IN_FEATURES", "128") or 128)
    out_features = int(os.environ.get("REVO_OUT_FEATURES", "64") or 64)
    rank = int(os.environ.get("REVO_RANK", "8") or 8)
    hidden_dim = int(os.environ.get("REVO_HIDDEN_DIM", "128") or 128)
    cfg = HyperLoraConfig(context_dim=ctx_dim, rank=rank, in_features=in_features, out_features=out_features, hidden_dim=hidden_dim)
    A, B, scale = hyperlora_generate(ctx, cfg)
    np.random.seed(42)
    W = np.random.standard_normal(size=(out_features, in_features)).astype(np.float32) * 0.02
    W2, handle = apply_delta(W, A, B, scale)
    x = np.random.standard_normal(size=(in_features,)).astype(np.float32)
    _ = W2 @ x
    W3 = revert_delta(W2, handle)
    assert np.allclose(W, W3, atol=1e-5)
    log_potential(mode_key, feats, ctx, scale)
    cache = ModeCache()
    cache.put(mode_key, ctx, signature={"rank": cfg.rank, "ctx_dim": cfg.context_dim})
    _ = cache.get(mode_key)
    output_text = "- Low-rank deltas are efficient.\n- Ephemeral application enables reversibility."
    log_identity_anchor(output_text, model="revo-cli")
    log_cognitive_conservation(output_text, model="revo-cli")
    print("OK: REVO run completed.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="Hola REVO! Give two bullets about ephemeral low-rank deltas.")
    args = p.parse_args()
    run(args.prompt)


if __name__ == "__main__":
    main()
