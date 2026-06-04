"""End-to-end NLL training of WeightLaw on wikitext-2.

Trains law to produce weight deltas that improve NLL. Uses wikitext-2
for diverse training data with held-out validation.

Usage:
    python train_law_e2e.py              # Default: wikitext, rank=16, 1000 steps
    python train_law_e2e.py quick        # 200 steps, 200 samples (test)
    python train_law_e2e.py rank=32      # Higher rank
    python train_law_e2e.py eval <ckpt>  # Generate with trained law
"""

import sys, time, os, glob
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.optim.lr_scheduler import CosineAnnealingLR

from revo.law_streaming import build_law
from revo._utils import load_model_tokenizer
from revo.streaming import (
    _get_device, _detect_arch, _n_layers, _d_model,
    _extract_shared, _non_layer_pattern,
    _get_layer_container, _arch_layer_prefix,
)


def load_wikitext(tokenizer, max_samples=2000, max_length=64):
    """Load wikitext-2, tokenize up to max_samples."""
    from datasets import load_dataset
    print("  Loading wikitext-2...")
    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize_split(ds, max_samp):
        texts = []
        for item in ds:
            text = item["text"].strip()
            if len(text) < 20:
                continue
            texts.append(text)
            if len(texts) >= max_samp:
                break
        return tokenizer(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).input_ids

    train_ids = tokenize_split(ds_train, max_samples)
    val_ids = tokenize_split(ds_val, min(max_samples // 4, 200))
    print(f"  Train: {len(train_ids)} seqs, Val: {len(val_ids)} seqs")
    return train_ids, val_ids


def train_e2e(model_name="HuggingFaceTB/SmolLM2-135M", rank=16,
              small_dim=8, hidden_dim=256, lr=3e-4, steps=1000,
              batch_size=2, max_samples=2000, max_length=64,
              val_every=50, val_samples=50, lambda_kl=0.0,
              do_generate=True):
    device = _get_device(load_model_tokenizer(model_name)[0])
    print(f"Device: {device}")

    model, tokenizer = load_model_tokenizer(model_name)
    model.to(device)
    model.train()

    d_model = _d_model(model)
    n_layers = _n_layers(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    print(f"Model: d_model={d_model}, n_layers={n_layers}, arch={arch}")

    # Build law
    law = build_law(model, rank=rank, small_dim=small_dim,
                    hidden_dim=hidden_dim,
                    cognitive=False, cognitive_field=False).to(device)
    law.train()
    total_params = sum(p.numel() for p in law.parameters())
    print(f"Law: {total_params:,} params ({total_params/135e6*100:.2f}% of model)")

    optimizer = torch.optim.AdamW(law.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    # Param name mapping
    param_map = {}
    for i in range(n_layers):
        lp = _arch_layer_prefix(i, arch)
        for pn, p in container[i].named_parameters():
            if p.dim() == 2:
                base = pn[:-7] if pn.endswith('.weight') and len(pn) > 7 else pn
                param_map[(i, base)] = f"{lp}{pn}"

    # Load data
    train_ids, val_ids = load_wikitext(tokenizer, max_samples, max_length)
    train_ids = train_ids.to(device)
    val_ids = val_ids.to(device)

    # Shared model params (frozen)
    non_layer = _non_layer_pattern(arch)
    state = model.state_dict(keep_vars=False)
    shared = {k: v for k, v in state.items() if not non_layer.search(k)}
    wte, wpe, _, _, _ = _extract_shared(shared, arch)
    wte_d = wte.to(device)
    wpe_d = wpe.to(device) if wpe is not None else None

    device_weights = {}
    for (i, b), fn in param_map.items():
        device_weights[(i, b)] = state[fn].to(device)

    # ── Baseline ──
    @torch.no_grad()
    def compute_nll(tokens):
        logits = model(tokens).logits
        sl = logits[:, :-1].reshape(-1, logits.shape[-1])
        return F.cross_entropy(sl, tokens[:, 1:].reshape(-1)).item()

    baseline_nll = compute_nll(val_ids[:val_samples])
    print(f"Baseline NLL (val): {baseline_nll:.4f}")

    model_short = model_name.split('/')[-1]
    save_path = f"checkpoints/law_e2e_{model_short}_r{rank}_ent.pt"

    # ── Training ──
    print(f"\n{'='*60}")
    print(f"Train: {steps} steps, lr={lr}, rank={rank}, batch={batch_size}")
    print(f"{'='*60}")

    # KL divergence: keep output close to original model (0 = off)
    # Delta magnitude penalty: prevent large weight modifications
    lambda_delta = 1e-3

    t0 = time.perf_counter()
    best_val_nll = float('inf')
    n_train = len(train_ids)

    for step in range(steps):
        optimizer.zero_grad()

        idx = (step * batch_size) % n_train
        batch = train_ids[idx: idx + batch_size]

        # Embed
        x = F.embedding(batch, wte_d)
        if wpe_d is not None:
            x = x + F.embedding(torch.arange(batch.shape[1], device=device), wpe_d)

        # Generate deltas for all layers
        total_delta_norm = 0.0
        n_deltas = 0
        overrides = {}
        for i in range(n_layers):
            gen = law(i)
            for bn, (u, vh) in gen.items():
                key = (i, bn)
                if key not in param_map:
                    continue
                delta = u @ vh
                total_delta_norm += delta.norm().item()
                n_deltas += 1
                wr = device_weights[key]
                overrides[param_map[key]] = wr.to(dtype=torch.float32) + delta.to(dtype=torch.float32)

        # Forward with law-generated weights
        logits_new = functional_call(model, overrides, (),
                                     {"inputs_embeds": x}).logits
        shift_logits = logits_new[:, :-1].reshape(-1, logits_new.shape[-1])
        shift_labels = batch[:, 1:].reshape(-1)

        # NLL loss
        loss_nll = F.cross_entropy(shift_logits, shift_labels)

        # KL divergence: keep new model close to original (optional)
        kl_val = 0.0
        if lambda_kl > 0:
            with torch.no_grad():
                logits_orig = model(inputs_embeds=x).logits
                shift_orig = logits_orig[:, :-1].reshape(-1, logits_orig.shape[-1])
            log_probs_new = F.log_softmax(shift_logits, dim=-1)
            probs_orig = F.softmax(shift_orig, dim=-1)
            kl_div = F.kl_div(log_probs_new, probs_orig, reduction='batchmean')
            kl_val = kl_div.item()
            loss_kl = lambda_kl * kl_div
        else:
            loss_kl = 0.0

        # Delta magnitude penalty
        loss_delta = lambda_delta * total_delta_norm / max(n_deltas, 1)

        loss = loss_nll + loss_kl + loss_delta
        loss.backward()
        torch.nn.utils.clip_grad_norm_(law.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % val_every == 0:
            # Validation
            law.eval()
            val_nll = 0.0
            with torch.no_grad():
                v_batch = val_ids[:val_samples]
                v_x = F.embedding(v_batch, wte_d)
                if wpe_d is not None:
                    v_x = v_x + F.embedding(
                        torch.arange(v_batch.shape[1], device=device), wpe_d)
                v_over = {}
                for i in range(n_layers):
                    gen = law(i)
                    for bn, (u, vh) in gen.items():
                        key = (i, bn)
                        if key not in param_map:
                            continue
                        wr = device_weights[key]
                        v_over[param_map[key]] = wr.to(dtype=torch.float32) + (u @ vh).to(dtype=torch.float32)
                v_logits = functional_call(model, v_over, (),
                                           {"inputs_embeds": v_x}).logits
                val_nll = F.cross_entropy(
                    v_logits[:, :-1].reshape(-1, v_logits.shape[-1]),
                    v_batch[:, 1:].reshape(-1)).item()
            law.train()

            train_nll = loss_nll.item()
            kl_val = kl_div.item() if isinstance(kl_div, torch.Tensor) else 0.0
            delta_norm = total_delta_norm / max(n_deltas, 1)
            delta = train_nll - baseline_nll
            val_delta = val_nll - baseline_nll
            elapsed = time.perf_counter() - t0

            saved = ""
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                law.cpu()
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(law.state_dict(), save_path.replace('.pt', '_best.pt'))
                law.to(device)
                saved = " ★ saved"

            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {step+1:4d}/{steps} | "
                  f"train={train_nll:.4f} (Δ={delta:+.3f}) | "
                  f"val={val_nll:.4f} (Δ={val_delta:+.3f}) | "
                    f"kl={kl_val:.4f} δ‖={delta_norm:.4f} | "
                  f"best={best_val_nll:.4f}{saved} | "
                  f"lr={lr_now:.2e} | {elapsed:.0f}s")

    total_time = time.perf_counter() - t0

    # ── Results ──
    print(f"\n{'='*60}")
    print(f"Baseline NLL: {baseline_nll:.4f}")
    print(f"Best val NLL: {best_val_nll:.4f} (Δ={best_val_nll - baseline_nll:+.4f})")
    print(f"Training: {total_time:.0f}s ({total_time/steps:.2f}s/step)")
    print(f"Law: {total_params:,} params")

    law.cpu()
    torch.save(law.state_dict(), save_path)
    print(f"Saved: {save_path}")
    print("Done.")
    return {
        "law_params": total_params,
        "best_val_nll": best_val_nll,
        "baseline_nll": baseline_nll,
        "nll_delta": best_val_nll - baseline_nll,
        "total_time_s": total_time,
        "checkpoint": save_path,
    }


def eval_law(checkpoint, model_name="HuggingFaceTB/SmolLM2-135M",
             prompt="The future of artificial intelligence", max_new=30):
    from revo.law_streaming import law_generate
    from revo._utils import load_model_tokenizer

    model, tokenizer = load_model_tokenizer(model_name)
    device = next(model.parameters()).device

    # Parse rank from filename
    parts = checkpoint.split('_')
    rank = 16
    for p in parts:
        if p.startswith('r'):
            try: rank = int(p[1:].split('.')[0])
            except: pass

    print(f"Loading checkpoint (rank={rank}): {checkpoint}")
    law = build_law(model, rank=rank, small_dim=min(rank//2, 8),
                    hidden_dim=256, cognitive=False, cognitive_field=False)
    sd = torch.load(checkpoint, map_location=device, weights_only=True)
    law.load_state_dict(sd)
    law.to(device)
    law.eval()

    prompts = [
        "The future of artificial intelligence",
        "Neural networks learn by",
        "Climate change poses",
        "The purpose of life is",
        "In the beginning",
        "The invention of",
        "Machine learning algorithms",
        "Scientists have discovered",
    ]

    print(f"\nPrompt: '{prompt}'")
    text, meta = law_generate(model, tokenizer, prompt, law=law,
                              max_new_tokens=max_new, temperature=0.8,
                              top_k=50, verbose=True)
    print(f"\nGenerated ({len(text)} chars):")
    print(text)
    print(f"\nSpeed: {meta['mean_time_per_token_s']*1000:.1f}ms/tok")

    # Also run with greedy decoding for comparison
    text_g, meta_g = law_generate(model, tokenizer, prompt, law=law,
                                   max_new_tokens=max_new, temperature=0.0,
                                   verbose=False)
    print(f"\nGreedy: {text_g[:200]}")
    print(f"Speed: {meta_g['mean_time_per_token_s']*1000:.1f}ms/tok")

    return text, meta


def parse_rank(s):
    for part in s.split(','):
        if '=' in part:
            k, v = part.split('=')
            if k.strip() == 'rank':
                return int(v.strip())
    return 16


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    if mode == "eval":
        ckpt = sys.argv[2] if len(sys.argv) > 2 else None
        if ckpt is None:
            ckpts = sorted(glob.glob("checkpoints/law_e2e_*best*"))
            ckpt = ckpts[-1] if ckpts else None
        if not ckpt:
            print("No checkpoint found.")
            sys.exit(1)
        print(f"Using checkpoint: {ckpt}")
        eval_law(ckpt)
    elif mode == "quick":
        train_e2e(steps=200, max_samples=200, val_every=50)
    elif mode.startswith("rank="):
        r = parse_rank(mode)
        train_e2e(rank=r, small_dim=max(4, r//2), steps=1000, max_samples=2000)
    else:
        train_e2e(steps=1000, max_samples=2000, val_every=50)
