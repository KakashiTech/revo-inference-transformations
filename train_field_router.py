"""Train the CognitiveField for selective weight generation.

Supports two modes:
  1. delta mode (default): law adds rank-4 deltas on real weights
  2. pure mode: law GENERATES all weights (rank=32, SVD pretrained)

Usage:
    python train_field_router.py tiny          # tiny-gpt2 quick test
    python train_field_router.py delta         # SmolLM2, rank-4 deltas (default)
    python train_field_router.py pure          # SmolLM2, rank=32, SVD pretrain + field
"""

import sys, time
import torch
import torch.nn.functional as F
from torch.func import functional_call

from revo.law_streaming import build_law, _extract_svd_targets, pretrain_law
from revo._utils import load_model_tokenizer
from revo.streaming import (
    _get_device, _detect_arch, _n_layers, _d_model,
    _extract_shared, _non_layer_pattern,
    _get_layer_container, _arch_layer_prefix,
)

TEXTS = [
    "The future of artificial intelligence will transform every aspect of human life.",
    "In the beginning the universe was created and this has made people very angry.",
    "Neural networks learn by adjusting their weights based on error signals.",
    "Machine learning focuses on learning patterns from data automatically.",
    "The complexity of deep learning models grows with each passing year.",
    "A transformer processes sequences using self-attention rather than recurrence.",
    "Training large language models requires vast amounts of text data.",
    "Understanding gradient descent is essential for deep learning practitioners.",
]


def train_field_router(model_name="sshleifer/tiny-gpt2", lr=0.001,
                       lambda_sparse=0.1, steps=100, batch_size=1,
                       rank=4, hidden_dim=64, small_dim=2,
                       pure=False, n_pretrain=0):
    """Train CognitiveField for selective weight generation.

    Args:
        pure: If True, law generates ALL weights (no W_real in functional_call).
              The law is SVD pretrained before field training.
        n_pretrain: Number of SVD pretrain steps (0 = skip).
    """
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_tokenizer(model_name)
    device = _get_device(model)
    model.to(device)
    model.train()

    d_model = _d_model(model)
    n_layers = _n_layers(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    print(f"Model: d_model={d_model}, n_layers={n_layers}, arch={arch}")
    print(f"Mode: {'PURE' if pure else 'DELTA'} rank={rank} small={small_dim}")

    # Build law (no ctx_proj — field only, cleaner signal)
    law = build_law(model, rank=rank, small_dim=small_dim, hidden_dim=hidden_dim,
                    cognitive=False, cognitive_field=True).to(device)

    # ── SVD pretraining ──
    if n_pretrain > 0:
        print(f"\nExtracting SVD targets (rank={rank})...")
        t0 = time.perf_counter()
        targets = _extract_svd_targets(model, rank=rank)
        print(f"  SVD done in {time.perf_counter()-t0:.1f}s "
              f"({len(targets)} layers)")

        print(f"Pre-training law via MSE ({n_pretrain} steps)...")
        losses = pretrain_law(law, targets, steps=n_pretrain, lr=1e-3,
                              device=device, verbose=True)
        print(f"  Final pretrain loss: {losses[-1]:.6f}")

    # ── Trainable: field only ──
    trainable = []
    for name, p in law.field.named_parameters():
        p.requires_grad = True
        trainable.append(p)
    trainable_ids = {id(p) for p in trainable}
    for p in law.parameters():
        if id(p) not in trainable_ids:
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_trainable:,} ({len(trainable)} tensors)")

    optimizer = torch.optim.Adam(trainable, lr=lr)

    # Build param name mapping: (layer_idx, weight_base_name) → full_model_param
    param_map = {}
    for i in range(n_layers):
        layer_mod_name = _arch_layer_prefix(i, arch)
        for pn, param in container[i].named_parameters():
            if param.dim() == 2:
                base = pn[:-7] if pn.endswith('.weight') and len(pn) > 7 else pn
                full_name = f"{layer_mod_name}{pn}"
                param_map[(i, base)] = full_name

    # Tokenize
    all_ids = tokenizer(TEXTS, return_tensors="pt", padding=True,
                        truncation=True, max_length=64).input_ids.to(device)

    # Shared params (embeddings, layernorms, lm_head)
    non_layer = _non_layer_pattern(arch)
    state = model.state_dict(keep_vars=False)
    shared = {k: v for k, v in state.items() if not non_layer.search(k)}
    wte, wpe, norm_w, norm_b, lm_head_w = _extract_shared(shared, arch)

    # ── Baseline: original model NLL ──
    print("\nComputing baseline NLL (original model)...")
    with torch.no_grad():
        baseline_ids = all_ids[:1]
        logits_base = model(baseline_ids).logits
        shift_logits = logits_base[:, :-1].reshape(-1, logits_base.shape[-1])
        shift_labels = baseline_ids[:, 1:].reshape(-1)
        baseline_nll = F.cross_entropy(shift_logits, shift_labels).item()
    print(f"  Original model NLL: {baseline_nll:.4f}")

    # ── Initial field state ──
    print("\nInitial field state:")
    with torch.no_grad():
        x = F.embedding(all_ids[:1], wte.to(device))
        if wpe is not None:
            x = x + F.embedding(
                torch.arange(all_ids.shape[1], device=device), wpe.to(device))
        fs_init = law.field(x)
        print(f"  {fs_init}")

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"Training: {steps} steps, lr={lr}, λ={lambda_sparse}")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    for step in range(steps):
        optimizer.zero_grad()

        # Sample batch (batch_size=1 for shape compatibility)
        batch_ids = all_ids[step % len(all_ids):step % len(all_ids) + batch_size]

        # Embed
        x = F.embedding(batch_ids, wte.to(device))
        if wpe is not None:
            x = x + F.embedding(
                torch.arange(batch_ids.shape[1], device=device), wpe.to(device))

        # Phase XII-c: compute field state from hidden state
        field_state = law.field(x)

        # Build parameter overrides per layer
        param_overrides = {}
        for i in range(n_layers):
            generated = law(i, hidden_state=x, field_state=field_state)
            for base_name, (U_eff, Vh) in generated.items():
                key = (i, base_name)
                if key not in param_map:
                    continue
                full_name = param_map[key]
                W = U_eff @ Vh
                if pure:
                    W_eff = W.to(dtype=torch.float32)
                else:
                    W_real = state[full_name].to(device)
                    W_eff = W_real.to(dtype=torch.float32) + W.to(dtype=torch.float32)
                param_overrides[full_name] = W_eff

        # Forward via functional_call
        fwd_kwargs = {"inputs_embeds": x}
        logits = functional_call(model, param_overrides, (), fwd_kwargs).logits

        # NLL loss
        shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
        shift_labels = batch_ids[:, 1:].reshape(-1)
        loss_nll = F.cross_entropy(shift_logits, shift_labels)

        # Sparsity loss
        loss_sparse = lambda_sparse * field_state._layer_scores.mean()
        loss = loss_nll + loss_sparse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        if (step + 1) % 10 == 0 or step == 0:
            with torch.no_grad():
                active_03 = field_state.layers_to_generate(threshold=0.3)
                active_05 = field_state.layers_to_generate(threshold=0.5)
                scores = field_state._layer_scores
            delta_nll = loss_nll.item() - baseline_nll
            elapsed = time.perf_counter() - t0
            print(f"  step {step+1:3d}/{steps} | "
                  f"NLL={loss_nll.item():.4f} (Δ={delta_nll:+.4f}) | "
                  f"sprs={loss_sparse.item():.4f} | "
                  f"act(>0.3)={len(active_03)}/{n_layers} "
                  f"act(>0.5)={len(active_05)}/{n_layers} | "
                  f"scores=[{scores.min().item():.2f}, "
                  f"{scores.mean().item():.2f}, "
                  f"{scores.max().item():.2f}] | "
                  f"{elapsed:.1f}s")

    total_time = time.perf_counter() - t0
    print(f"\n{'='*60}")
    with torch.no_grad():
        x_final = F.embedding(all_ids[:1], wte.to(device))
        if wpe is not None:
            x_final = x_final + F.embedding(
                torch.arange(all_ids.shape[1], device=device), wpe.to(device))
        fs = law.field(x_final)
        print(f"Final field: {fs}")
        for th in [0.3, 0.5, 0.7]:
            active = fs.layers_to_generate(threshold=th)
            print(f"  active(>{th}): {len(active)}/{n_layers}")
        print(f"  scores: [{fs._layer_scores.min().item():.3f}, "
              f"{fs._layer_scores.mean().item():.3f}, "
              f"{fs._layer_scores.max().item():.3f}]")

        logits_final = functional_call(model, param_overrides, (),
                                        {"inputs_embeds": x_final}).logits
        shift_logits = logits_final[:, :-1].reshape(-1, logits_final.shape[-1])
        shift_labels = all_ids[:1, 1:].reshape(-1)
        final_nll = F.cross_entropy(shift_logits, shift_labels).item()
    print(f"\nNLL: baseline={baseline_nll:.4f} final={final_nll:.4f} "
          f"Δ={final_nll - baseline_nll:+.4f}")
    print(f"Training time: {total_time:.1f}s")
    print("Done.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "delta"
    if mode == "tiny":
        train_field_router("sshleifer/tiny-gpt2", lr=0.005,
                          lambda_sparse=0.05, steps=60, batch_size=1,
                          rank=4, hidden_dim=64)
    elif mode == "pure":
        # Pure mode: rank=32, small=16 for output expressivity
        # SVD pretrain then field training
        train_field_router("HuggingFaceTB/SmolLM2-135M", lr=0.001,
                          lambda_sparse=0.2, steps=300, batch_size=1,
                          rank=32, hidden_dim=256, small_dim=16,
                          pure=True, n_pretrain=500)
    else:
        # Delta mode: rank=4, field training on deltas
        train_field_router("HuggingFaceTB/SmolLM2-135M", lr=0.001,
                          lambda_sparse=0.3, steps=300, batch_size=1,
                          rank=4, hidden_dim=64, pure=False, n_pretrain=0)
