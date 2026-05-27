from __future__ import annotations

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo.features import build_features, build_context_vector, compute_mode_key
from revo.hyperlora import HyperLoraConfig, hyperlora_generate
from revo.mode_cache import ModeCache
from revo.potentials import log_potential
from revo.observability import log_identity_anchor, log_cognitive_conservation
from revo.torch_integration import apply_delta_torch, revert_delta_torch


def _pick_linear_module(model: torch.nn.Module) -> torch.nn.Linear:
    try:
        return model.transformer.h[-1].mlp.c_fc  # type: ignore[attr-defined]
    except Exception:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.weight.dim() == 2:
                return m
        raise RuntimeError("No suitable Linear module found")


def main() -> None:
    seed = int(os.environ.get("REVO_SEED", "0") or 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_name = os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2")
    prompt = os.environ.get("REVO_PROMPT", "Hola REVO! Dame dos bullets sobre deltas low-rank.")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    target = _pick_linear_module(model)
    of, inf = target.weight.shape

    feats = build_features(prompt)
    mode_key = compute_mode_key(feats, prompt)

    ctx_dim = int(os.environ.get("REVO_CTX_DIM", "64") or 64)
    ctx = build_context_vector(feats, context_dim=ctx_dim, seed=seed)

    rank = int(os.environ.get("REVO_RANK", "8") or 8)
    hidden_dim = int(os.environ.get("REVO_HIDDEN_DIM", "128") or 128)
    cfg = HyperLoraConfig(context_dim=ctx_dim, rank=rank, in_features=int(inf), out_features=int(of), hidden_dim=hidden_dim)

    A, B, scale = hyperlora_generate(ctx, cfg)

    W0 = target.weight.detach().clone()
    handle = apply_delta_torch(target, A, B, scale)

    tokens = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**tokens, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)

    revert_delta_torch(handle)
    assert torch.allclose(target.weight, W0, atol=1e-6), "Reversibility check failed for torch module"

    log_potential(mode_key, feats, ctx, scale)
    cache = ModeCache()
    cache.put(mode_key, ctx, signature={"rank": cfg.rank, "ctx_dim": cfg.context_dim})
    _ = cache.get(mode_key)
    log_identity_anchor(text, model=model_name)
    log_cognitive_conservation(text, model=model_name)

    print("OK: REVO LLM smoke completed.")
    print(text)


if __name__ == "__main__":
    main()
