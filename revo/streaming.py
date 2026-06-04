"""Streaming — inference with per-layer weight streaming from disk.

Never keep more than one transformer block's weights in RAM at a time.
This is the infrastructure that makes REVO's core claim real:
    "El modelo no vive en la RAM; vive en el tiempo."

Usage:
    from revo.streaming import shard_model, stream_forward, stream_generate

    # 1. Shard weights to disk (one-time per model)
    info = shard_model(model, "./shards/gpt2")

    # 2. Stream forward (only ~1/N weights in RAM at peak)
    logits = stream_forward(model, input_ids, "./shards/gpt2")

    # 3. Generate with streaming
    tokens = stream_generate(model, tokenizer, "Hello", "./shards/gpt2", max_new=20)

    # 4. Compare memory
    report = compare_memory(model, tokenizer, "./shards/gpt2", texts)
"""

from __future__ import annotations

import gc
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from revo._utils import measure_memory_rss, free_memory_trim


def _layer_pattern(layer_idx: int, arch: str) -> re.Pattern:
    if arch == "gpt2":
        return re.compile(re.escape(f"transformer.h.{layer_idx}."))
    if arch == "llama":
        return re.compile(re.escape(f"model.layers.{layer_idx}."))
    # Fallback: match any pattern that contains .layer_idx.
    return re.compile(r'(?:transformer\.h\.|model\.layers\.)' + re.escape(str(layer_idx)) + r'\.')


def _non_layer_pattern(arch: str) -> re.Pattern:
    if arch == "gpt2":
        return re.compile(r'transformer\.h\.\d+\.')
    if arch == "llama":
        return re.compile(r'model\.layers\.\d+\.')
    return re.compile(r'(?:transformer\.h\.|model\.layers\.)\d+\.')


def shard_model(model: nn.Module, output_dir: str) -> Dict[str, int]:
    """Save model weights sharded by layer.

    Auto-detects architecture (GPT-2, LLaMA/Qwen2).

    Produces:
        output_dir/shared.pt      — embedding, final norm, lm_head
        output_dir/layer_0.pt     — transformer block 0
        output_dir/layer_1.pt     — transformer block 1
        ...

    Returns {shard_name: bytes} mapping.
    """
    os.makedirs(output_dir, exist_ok=True)
    state = model.state_dict()
    arch = _detect_arch(model)
    n_layers = _n_layers(model)
    info: Dict[str, int] = {}

    non_layer = _non_layer_pattern(arch)
    shared = {k: v for k, v in state.items() if not non_layer.search(k)}
    _save_shard(output_dir, "shared", shared, info)

    for i in range(n_layers):
        pat = _layer_pattern(i, arch)
        layer = {k: v for k, v in state.items() if pat.search(k)}
        _save_shard(output_dir, f"layer_{i}", layer, info)

    try:
        model.config.save_pretrained(output_dir)
    except Exception:
        pass

    free_memory_trim()
    return info


def _save_shard(output_dir: str, name: str,
                tensors: Dict[str, torch.Tensor],
                info: Dict[str, int]) -> None:
    path = os.path.join(output_dir, f"{name}.pt")
    torch.save(tensors, path)
    total = sum(v.numel() * v.element_size() for v in tensors.values())
    info[name] = total


_ARCH_PATTERNS: Dict[str, Any] = {}


def _detect_arch(model: nn.Module) -> str:
    """Detect model architecture."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    return "unknown"


def _get_layer_container(model: nn.Module):
    arch = _detect_arch(model)
    if arch == "gpt2":
        return model.transformer.h
    if arch == "llama":
        return model.model.layers
    raise ValueError(f"Cannot find layer container for arch={arch}")


def _get_block_keys(layer_idx: int) -> Tuple[str, str]:
    """Return (shard_key_prefix, state_dict_prefix) for a given layer."""
    return f"transformer.h.{layer_idx}.", f"model.layers.{layer_idx}."


def _n_layers(model: nn.Module) -> int:
    if hasattr(model, "config") and hasattr(model.config, "n_layer"):
        return int(model.config.n_layer)
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    try:
        return len(_get_layer_container(model))
    except (ValueError, AttributeError):
        pass
    raise ValueError("Cannot determine number of layers")


def _d_model(model: nn.Module) -> int:
    if hasattr(model, "config") and hasattr(model.config, "n_embd"):
        return int(model.config.n_embd)
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    raise ValueError("Cannot determine model dimension")


def _arch_layer_prefix(layer_idx: int, arch: str) -> str:
    if arch == "gpt2":
        return f"transformer.h.{layer_idx}."
    if arch == "llama":
        return f"model.layers.{layer_idx}."
    return ""


def _get_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _free_module_weights(module: nn.Module) -> None:
    """Set all parameters of a module to empty tensors (free memory)."""
    for p in module.parameters():
        p.data = torch.empty(0, dtype=p.dtype, device=p.device)


def _free_module_2d_weights(module: nn.Module) -> int:
    """Free only 2D weight matrices (leave biases, layernorms intact).
    
    Sets freed weights to shape (0, 0) so dim() == 2 is preserved.
    Returns count of params freed.
    """
    n = 0
    for p in module.parameters():
        if p.dim() == 2 and p.numel() > 0:
            p.data = torch.empty(0, 0, dtype=p.dtype, device=p.device)
            n += 1
    return n


def _free_block_2d_weights(block: nn.Module) -> int:
    """Free a transformer block's 2D weights only."""
    return _free_module_2d_weights(block)


def _free_all_block_2d_weights(model: nn.Module) -> int:
    """Free all 2D block weights. Returns count of freed params."""
    n = 0
    for block in _get_layer_container(model):
        n += _free_block_2d_weights(block)
    free_memory_trim()
    return n


def _load_block_from_shard(block: nn.Module, shard_dir: str,
                            layer_idx: int, device: torch.device,
                            arch: str = "gpt2") -> None:
    """Load a transformer block's weights from disk into its parameters."""
    path = os.path.join(shard_dir, f"layer_{layer_idx}.pt")
    weights = torch.load(path, map_location=device, weights_only=True)
    prefix = _arch_layer_prefix(layer_idx, arch)
    for name, param in block.named_parameters():
        key = f"{prefix}{name}"
        if key in weights:
            param.data = weights[key].to(device=device, dtype=param.dtype)


def _free_block_weights(block: nn.Module) -> None:
    """Free a transformer block's weights."""
    _free_module_weights(block)


def _free_all_block_weights(model: nn.Module) -> int:
    """Free all transformer block weights. Returns count of blocks freed."""
    n = 0
    for block in _get_layer_container(model):
        _free_block_weights(block)
        n += 1
    free_memory_trim()
    return n


def _restore_all_blocks(model: nn.Module, shard_dir: str) -> int:
    """Restore all block weights from disk. Returns count of blocks restored."""
    device = _get_device(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    for i in range(len(container)):
        _load_block_from_shard(container[i], shard_dir, i, device, arch)
    return len(container)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm (used by LLaMA/Qwen2). No bias, no mean subtraction."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(dtype)


def _shared_keys(arch: str) -> Tuple[str, str, str, str, str]:
    """Return (embed_key, pos_embed_key, norm_w_key, norm_b_key, lm_head_key)."""
    if arch == "gpt2":
        return ("transformer.wte.weight", "transformer.wpe.weight",
                "transformer.ln_f.weight", "transformer.ln_f.bias",
                "lm_head.weight")
    if arch == "llama":
        return ("model.embed_tokens.weight", None,
                "model.norm.weight", None,
                "lm_head.weight")
    raise ValueError(f"Unknown arch: {arch}")


def _extract_shared(shared: Dict[str, torch.Tensor], arch: str) -> Tuple:
    ek, pk, nwk, nbk, lhk = _shared_keys(arch)
    wte = shared[ek]
    wpe = shared.get(pk) if pk else None
    norm_w = shared.get(nwk)
    norm_b = shared.get(nbk)
    lm_w = shared.get(lhk, shared.get(ek))
    return wte, wpe, norm_w, norm_b, lm_w


def _get_rotary_emb(model: nn.Module) -> Optional[Any]:
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    return None


def _build_block_kwargs(arch: str, x: torch.Tensor,
                         position_ids: torch.Tensor,
                         rotary_emb: Optional[Any] = None) -> Dict[str, Any]:
    """Build architecture-specific kwargs for block forward."""
    kwargs: Dict[str, Any] = {}
    if arch == "llama" and rotary_emb is not None:
        if position_ids.dim() == 1:
            pid = position_ids.unsqueeze(0)
        else:
            pid = position_ids
        cos, sin = rotary_emb(x, pid)
        kwargs["position_embeddings"] = (cos, sin)
    return kwargs


@torch.no_grad()
def stream_forward(model: nn.Module, input_ids: torch.Tensor,
                   shard_dir: str,
                   use_cache: bool = True,
                   past_key_values: Optional[Any] = None,
                   output_attentions: bool = False,
                   restore: bool = True) -> torch.Tensor:
    """Single forward pass with per-layer weight streaming.

    Peak memory: shared weights + ONE transformer block + KV cache.
    Block weights are freed before and (optionally) restored after.

    Supports GPT-2 and LLaMA/Qwen2 architectures.

    Args:
        model: The model object (architecture, weights streamed from disk)
        input_ids: (1, seq_len) token IDs
        shard_dir: Directory containing sharded weights
        use_cache: Whether to maintain KV cache
        past_key_values: Optional cached past key-values
        output_attentions: Not supported in streaming mode
        restore: If True, restore all block weights after forward

    Returns:
        logits: (1, seq_len, vocab_size)
    """
    device = _get_device(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    n_layers = len(container)
    seq_len = input_ids.shape[1]

    _free_all_block_weights(model)

    shared = torch.load(
        os.path.join(shard_dir, "shared.pt"),
        map_location=device, weights_only=True,
    )
    wte, wpe, norm_w, norm_b, lm_head_w = _extract_shared(shared, arch)

    try:
        position_ids_1d = torch.arange(seq_len, device=device, dtype=torch.long)
        position_ids = position_ids_1d.unsqueeze(0)
        x = F.embedding(input_ids, wte)
        if wpe is not None:
            x = x + F.embedding(position_ids_1d, wpe)

        rotary_emb = _get_rotary_emb(model)

        if use_cache:
            presents: List[Any] = []

        for i in range(n_layers):
            _load_block_from_shard(
                container[i], shard_dir, i, device, arch
            )

            pkv = past_key_values[i] if (use_cache and past_key_values is not None and i < len(past_key_values)) else None
            block_kwargs = _build_block_kwargs(arch, x, position_ids, rotary_emb)
            out = container[i](
                x,
                past_key_value=pkv,
                use_cache=use_cache,
                output_attentions=False,
                **block_kwargs,
            )
            if use_cache and isinstance(out, tuple) and len(out) >= 2:
                x = out[0]
                presents.append(out[1])
            else:
                x = out[0] if isinstance(out, tuple) else out

            _free_block_weights(container[i])

        if norm_w is not None:
            if arch == "llama":
                x = _rms_norm(x, norm_w, getattr(model.config, "rms_norm_eps", 1e-6))
            else:
                x = F.layer_norm(x, (x.shape[-1],), norm_w, norm_b)
        logits = F.linear(x, lm_head_w)

        return logits

    finally:
        if restore:
            _restore_all_blocks(model, shard_dir)


@torch.no_grad()
def stream_generate(model: nn.Module, tokenizer: Any,
                    prompt: str, shard_dir: str,
                    max_new_tokens: int = 20,
                    temperature: float = 1.0,
                    top_k: Optional[int] = None,
                    top_p: Optional[float] = None,
                    verbose: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Generate text with streaming weights.

    Each token requires a full forward pass through all layers,
    with weights loaded and discarded per layer per token.

    Returns:
        (generated_text, metadata)
    """
    device = _get_device(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    n_layers = len(container)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    _free_all_block_weights(model)

    shared = torch.load(
        os.path.join(shard_dir, "shared.pt"),
        map_location=device, weights_only=True,
    )
    wte, wpe, norm_w, norm_b, lm_head_w = _extract_shared(shared, arch)
    rotary_emb = _get_rotary_emb(model)

    generated = input_ids.clone()
    past_key_values: Optional[List[Any]] = None
    total_time = 0.0
    n_new = 0

    for t in range(max_new_tokens):
        t0 = time.perf_counter()
        cur_len = generated.shape[1]
        position_ids_1d = torch.arange(cur_len, device=device, dtype=torch.long)
        position_ids = position_ids_1d.unsqueeze(0)
        x = F.embedding(generated, wte)
        if wpe is not None:
            x = x + F.embedding(position_ids_1d, wpe)

        current_past = past_key_values or [None] * n_layers
        use_kv = past_key_values is not None

        for i in range(n_layers):
            _load_block_from_shard(
                container[i], shard_dir, i, device, arch
            )
            pkv = current_past[i] if use_kv else None
            inp = x[:, -1:] if use_kv else x
            block_pos_ids = position_ids[:, -1:] if use_kv else position_ids
            block_kwargs = _build_block_kwargs(arch, inp, block_pos_ids, rotary_emb)
            out = container[i](
                inp,
                past_key_value=pkv,
                use_cache=True,
                output_attentions=False,
                **block_kwargs,
            )
            if isinstance(out, tuple):
                x = out[0]
                if len(out) >= 2:
                    current_past[i] = out[1]
            else:
                x = out
            _free_block_weights(container[i])

        past_key_values = current_past
        x_last = x[:, -1]
        if norm_w is not None:
            if arch == "llama":
                x_last = _rms_norm(x_last, norm_w, getattr(model.config, "rms_norm_eps", 1e-6))
            else:
                x_last = F.layer_norm(x_last, (x_last.shape[-1],), norm_w, norm_b)
        logits = F.linear(x_last, lm_head_w)

        if temperature not in (0.0, 1.0):
            logits = logits / temperature
        if top_k is not None:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = float("-inf")
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        n_new += 1

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if verbose:
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(f"  [{t+1}/{max_new_tokens}] {token_str} ({elapsed*1000:.0f}ms/tok)")

        if next_token.item() == tokenizer.eos_token_id:
            break

    _restore_all_blocks(model, shard_dir)

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    meta = {
        "n_layers": n_layers,
        "total_time_s": total_time,
        "mean_time_per_token_s": total_time / max(n_new, 1),
        "n_new_tokens": n_new,
    }
    return generated_text, meta


@torch.no_grad()
def measure_streaming_memory(model: nn.Module, tokenizer: Any,
                              shard_dir: str, texts: List[str],
                              max_length: int = 128,
                              n_warmup: int = 2) -> Dict[str, Any]:
    """Measure peak memory of streaming forward vs full load.

    Returns dict with:
        - full_rss_bytes: RSS after full model load
        - streamed_peak_rss_bytes: peak RSS during streaming forward
        - full_rss_mb, streamed_peak_rss_mb: same in MB
        - n_layers: number of transformer layers
        - layer_params_mb: approximate MB per layer
    """
    rss_full = measure_memory_rss()

    free_memory_trim()
    _free_all_block_weights(model)
    free_memory_trim()
    rss_without_blocks = measure_memory_rss()

    # Run a streaming forward and track peak RSS
    rss_peak = 0
    for text in texts[:n_warmup]:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length)
        ids = enc.input_ids.to(_get_device(model))
        stream_forward(model, ids, shard_dir)
        rss_now = measure_memory_rss()
        rss_peak = max(rss_peak, rss_now)

    _restore_all_blocks(model, shard_dir)

    return {
        "full_rss_bytes": rss_full,
        "without_blocks_rss_bytes": rss_without_blocks,
        "streamed_peak_rss_bytes": rss_peak,
        "full_rss_mb": rss_full / 1024 / 1024,
        "without_blocks_rss_mb": rss_without_blocks / 1024 / 1024,
        "streamed_peak_rss_mb": rss_peak / 1024 / 1024,
        "n_layers": _n_layers(model),
        "d_model": _d_model(model),
    }


def _compute_shard_sizes(shard_dir: str) -> Dict[str, int]:
    """Compute tensor memory (bytes) per shard without loading."""
    sizes: Dict[str, int] = {}
    for fname in os.listdir(shard_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(shard_dir, fname)
        w = torch.load(path, map_location="cpu", weights_only=True)
        total = sum(v.numel() * v.element_size() for v in w.values())
        sizes[fname.replace(".pt", "")] = total
        del w
    return sizes


def compare_memory(model: nn.Module, tokenizer: Any,
                   shard_dir: str, texts: List[str],
                   max_length: int = 128) -> Dict[str, Any]:
    """Compare full load vs streaming memory and performance.

    Reports both theoretical tensor memory and measured RSS peak.
    The theoretical savings are guaranteed: at peak, only shared + 1 layer
    worth of weight tensors are resident.

    Returns comprehensive report.
    """
    device = _get_device(model)
    n_layers = _n_layers(model)
    result: Dict[str, Any] = {}

    # Theoretical tensor memory (exact, not noisy)
    shard_sizes = _compute_shard_sizes(shard_dir)
    shared_bytes = shard_sizes.get("shared", 0)
    layer_bytes = [v for k, v in shard_sizes.items() if k.startswith("layer_")]
    full_weight_bytes = shared_bytes + sum(layer_bytes)
    stream_peak_bytes = shared_bytes + (max(layer_bytes) if layer_bytes else 0)

    result["theory"] = {
        "shared_mb": shared_bytes / 1024 / 1024,
        "per_layer_mb": [b / 1024 / 1024 for b in layer_bytes],
        "full_weights_mb": full_weight_bytes / 1024 / 1024,
        "stream_peak_weights_mb": stream_peak_bytes / 1024 / 1024,
        "savings_pct": (1 - stream_peak_bytes / max(full_weight_bytes, 1)) * 100,
        "n_layers": n_layers,
    }

    # --- Phase 1: Full load baseline ---
    free_memory_trim()
    gc.collect()
    rss_before = measure_memory_rss()
    t0 = time.perf_counter()
    nll_full = _eval_nll_full(model, tokenizer, texts, max_length)
    t_full = time.perf_counter() - t0
    rss_full = measure_memory_rss()

    result["full"] = {
        "nll": nll_full,
        "time_s": t_full,
        "rss_mb": rss_full / 1024 / 1024,
        "rss_delta_mb": (rss_full - rss_before) / 1024 / 1024,
    }

    # --- Phase 2: Streaming (measured with peak tracking) ---
    free_memory_trim()
    _free_all_block_weights(model)
    free_memory_trim()
    rss_bare = measure_memory_rss()

    # Peak RSS tracking during streaming
    rss_peak = rss_bare
    def _track_rss():
        nonlocal rss_peak
        rss_peak = max(rss_peak, measure_memory_rss())

    t0 = time.perf_counter()
    nll_stream = _eval_nll_streaming(model, tokenizer, texts, shard_dir,
                                     max_length, _track_fn=_track_rss)
    t_stream = time.perf_counter() - t0
    rss_after = measure_memory_rss()

    result["streaming"] = {
        "nll": nll_stream,
        "nll_delta": nll_stream - nll_full,
        "time_s": t_stream,
        "time_ratio": t_stream / max(t_full, 1e-9),
        "rss_bare_mb": rss_bare / 1024 / 1024,
        "rss_peak_mb": rss_peak / 1024 / 1024,
        "rss_final_mb": rss_after / 1024 / 1024,
        "rss_peak_delta_mb": (rss_peak - rss_bare) / 1024 / 1024,
    }

    _restore_all_blocks(model, shard_dir)

    result["model"] = {
        "n_layers": n_layers,
        "d_model": _d_model(model),
        "total_params": sum(p.numel() for p in model.parameters()),
    }

    return result


def _eval_nll_full(model: nn.Module, tokenizer: Any,
                   texts: List[str], max_length: int) -> float:
    device = _get_device(model)
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length)
        input_ids = enc.input_ids.to(device)
        out = model(input_ids, labels=input_ids)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        n_tok = shift_labels.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


def _eval_nll_streaming(model: nn.Module, tokenizer: Any,
                        texts: List[str], shard_dir: str,
                        max_length: int,
                        _track_fn: Optional[Any] = None) -> float:
    device = _get_device(model)
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        if _track_fn:
            _track_fn()
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length)
        input_ids = enc.input_ids.to(device)
        logits = stream_forward(model, input_ids, shard_dir, restore=False)
        if _track_fn:
            _track_fn()
        if logits is None:
            continue
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        n_tok = shift_labels.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


# ========================================================================
# Phase XII-b: SVD Streaming — weights as reconstructed phenomenon
# "El modelo no es un tensor: es una ley de transformación reproducible."
# ========================================================================


@torch.no_grad()
def svd_shard_model(model: nn.Module, output_dir: str, rank: int = 64) -> Dict[str, Any]:
    """Save SVD-compressed weights per layer: W ≈ U_eff @ Vh.

    Decomposes every 2D weight matrix in each transformer block using SVD in
    the native storage frame, so reconstruction shape always matches the
    original parameter — no Conv1D/Linear orientation bookkeeping needed.

    Produces:
        output_dir/shared.pt       — tiny params (layernorms, lm_head, embedding)
        output_dir/layer_0.pt      — SVD factors for block 0
        output_dir/layer_{i}.pt    — SVD factors for block i
        output_dir/embedding.bin   — wte as raw float32 for memmap row-loading

    Returns info dict with compression statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    state = model.state_dict()
    arch = _detect_arch(model)
    n_layers = _n_layers(model)
    d = _d_model(model)
    r = max(1, min(rank, d))
    info: Dict[str, Any] = {"rank": r, "layers": n_layers, "arch": arch}

    non_layer = _non_layer_pattern(arch)
    shared = {k: v.clone() for k, v in state.items() if not non_layer.search(k)}
    _save_shard(output_dir, "shared", shared, info)

    total_orig = 0
    total_svd = 0

    for i in range(n_layers):
        pat = _layer_pattern(i, arch)
        layer_items = [(k, v) for k, v in state.items() if pat.search(k)]
        svd_factors: Dict[str, torch.Tensor] = {}

        arch_prefix = _arch_layer_prefix(i, arch)
        prefix_len = len(arch_prefix)

        for key, tensor in layer_items:
            relative = key[prefix_len:] if prefix_len and key.startswith(arch_prefix) else key
            if tensor.dim() < 2:
                svd_factors[relative] = tensor.cpu()
                continue

            m, n = tensor.shape
            ri = min(r, m, n)
            W = tensor.float().cpu()

            if ri * (m + n) * 4 < m * n * 4:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                U_eff = U[:, :ri] * S[:ri].unsqueeze(0)
                Vh_trunc = Vh[:ri, :].contiguous()
                base = relative[:-7] if relative.endswith('.weight') and len(relative) > 7 else relative
                svd_factors[f"{base}.U_eff"] = U_eff
                svd_factors[f"{base}.Vh"] = Vh_trunc

                total_orig += m * n * 4
                total_svd += (m * ri + ri * n) * 4
            else:
                svd_factors[relative] = tensor.cpu()

        _save_shard(output_dir, f"layer_{i}", svd_factors, info)

    # Save embedding as raw float32 bin for memmap row-loading
    ek, *extra = _shared_keys(arch)
    if ek in shared:
        wte = shared[ek].float().cpu().numpy()
        wte.tofile(os.path.join(output_dir, "embedding.bin"))
        info["embedding_shape"] = list(wte.shape)

    info["total_original_bytes"] = total_orig
    info["total_svd_bytes"] = total_svd
    info["compression_ratio"] = total_orig / max(total_svd, 1)

    free_memory_trim()
    return info


@torch.no_grad()
def _reconstruct_block_svd(block: nn.Module, svd_factors: Dict[str, torch.Tensor],
                            device: torch.device) -> None:
    """Reconstruct block weights from SVD factors (U_eff @ Vh) in-place.

    SVD factor keys use relative param names (e.g., 'attn.c_attn.U_eff').
    For 1D params (biases, layernorms), the full relative name is the key.
    Since weights are freed before reconstruction, param.data has shape (0,)
    — we assign directly without shape checks.
    """
    for name, param in block.named_parameters():
        base = name[:-7] if name.endswith('.weight') and len(name) > 7 else name
        u_key = f"{base}.U_eff"
        v_key = f"{base}.Vh"

        if u_key in svd_factors and v_key in svd_factors:
            U_eff = svd_factors[u_key].to(device=device, dtype=param.dtype)
            Vh = svd_factors[v_key].to(device=device, dtype=param.dtype)
            W = U_eff @ Vh
            param.data = W
        elif name in svd_factors:
            t = svd_factors[name].to(device=device, dtype=param.dtype)
            param.data = t


@torch.no_grad()
def _load_block_from_svd(block: nn.Module, svd_dir: str,
                          layer_idx: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load SVD factors and reconstruct block weights in-place.

    Returns the SVD factors dict for delta generation.
    """
    path = os.path.join(svd_dir, f"layer_{layer_idx}.pt")
    svd_factors = torch.load(path, map_location="cpu", weights_only=True)
    _reconstruct_block_svd(block, svd_factors, device)
    return svd_factors


@torch.no_grad()
def _apply_hyperlora_deltas(block: nn.Module,
                             svd_factors: Dict[str, torch.Tensor],
                             hidden_state: torch.Tensor,
                             device: torch.device,
                             delta_rank: int = 4,
                             context_dim: Optional[int] = None) -> None:
    """Generate and apply generative-law deltas on top of SVD-reconstructed weights.

    For each weight matrix in the block, generates a low-rank delta from the
    hidden state (context vector) using HyperLoRA. The delta is ADDED to the
    reconstructed SVD weight, making the effective weights input-dependent.

    This is the core of the paradigm shift: weights are not static —
    they are generated from a law conditioned on the input.
    """
    from revo.hyperlora import hyperlora_generate, HyperLoraConfig

    ctx_np = hidden_state.float().mean(dim=(0, 1)).cpu().numpy()
    d_ctx = context_dim or ctx_np.shape[0]

    for name, param in block.named_parameters():
        base = name[:-7] if name.endswith('.weight') and len(name) > 7 else name
        u_key = f"{base}.U_eff"
        v_key = f"{base}.Vh"

        if u_key not in svd_factors or v_key not in svd_factors:
            continue

        U_eff = svd_factors[u_key]  # (out, r)
        Vh = svd_factors[v_key]     # (r, in)

        U_shape = U_eff.shape  # (out, r)
        V_shape = Vh.shape     # (r, in)

        out_f, r_svd = U_shape
        r_svd2, in_f = V_shape

        cfg = HyperLoraConfig(
            context_dim=d_ctx,
            rank=delta_rank,
            in_features=in_f,
            out_features=out_f,
            hidden_dim=min(128, 2 * d_ctx),
            seed=int(hash(base) & 0x7FFFFFFF),
        )

        A_np, B_np, scale = hyperlora_generate(ctx_np, cfg)
        A = torch.from_numpy(A_np).to(device=device, dtype=param.dtype)   # (r, in_f)
        B = torch.from_numpy(B_np).to(device=device, dtype=param.dtype)   # (out_f, r)
        delta = (B @ A) * scale                                           # (out_f, in_f)

        # Add delta to the existing weight (already reconstructed from SVD)
        if param.data.shape == delta.shape:
            param.data.add_(delta)
        else:
            param.data.add_(delta.T)


@torch.no_grad()
def svd_stream_forward(model: nn.Module, input_ids: torch.Tensor,
                        svd_dir: str,
                        rank: int = 64,
                        use_cache: bool = True,
                        past_key_values=None,
                        memmap_embedding: bool = False,
                        output_attentions: bool = False,
                        restore: bool = True,
                        use_deltas: bool = False,
                        delta_rank: int = 4) -> torch.Tensor:
    """Forward pass with weights reconstructed on-the-fly from SVD bases.

    Each layer's weight matrices exist only as compact SVD factors (U_eff, Vh)
    on disk. During forward, they are loaded, reconstructed via a single matmul
    (U_eff @ Vh), assigned to the block parameters, used for the forward pass,
    then freed. Peak memory: shared weights + ONE reconstructed layer's weights
    + SVD factors of ONE layer.

    With use_deltas=True, applies a generative-law delta (HyperLoRA) on top of
    the SVD-reconstructed weights, making the effective weights input-dependent.
    This is the paradigm shift: "El modelo no es un tensor: es una ley de
    transformación reproducible."

    With memmap_embedding=True, the embedding table is never fully loaded into
    RAM — rows are paged in on demand from a raw binary file via numpy memmap,
    eliminating the largest persistent memory consumer.

    Returns logits: (1, seq_len, vocab_size).
    """
    device = _get_device(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    n_layers = len(container)

    _free_all_block_weights(model)

    shared = torch.load(
        os.path.join(svd_dir, "shared.pt"),
        map_location=device, weights_only=True,
    )
    wte, wpe, norm_w, norm_b, lm_head_w = _extract_shared(shared, arch)

    # If memmap_embedding, replace wte with memmap-backed tensor (row-on-demand)
    if memmap_embedding:
        bin_path = os.path.join(svd_dir, "embedding.bin")
        if os.path.exists(bin_path) and wte is not None:
            shape = wte.shape
            mm = np.memmap(bin_path, dtype=np.float32, mode="r", shape=shape)
            wte = torch.from_numpy(mm)

    try:
        seq_len = input_ids.shape[1]
        position_ids_1d = torch.arange(seq_len, device=device, dtype=torch.long)
        position_ids = position_ids_1d.unsqueeze(0)
        x = F.embedding(input_ids, wte)
        if wpe is not None:
            x = x + F.embedding(position_ids_1d, wpe)

        rotary_emb = _get_rotary_emb(model)

        if use_cache:
            presents = []

        for i in range(n_layers):
            svd_factors = _load_block_from_svd(container[i], svd_dir, i, device)

            if use_deltas:
                _apply_hyperlora_deltas(
                    container[i], svd_factors, x, device,
                    delta_rank=delta_rank,
                    context_dim=x.shape[-1],
                )

            pkv = past_key_values[i] if (use_cache and past_key_values is not None
                                          and i < len(past_key_values)) else None
            block_kwargs = _build_block_kwargs(arch, x, position_ids, rotary_emb)
            out = container[i](
                x,
                past_key_value=pkv,
                use_cache=use_cache,
                output_attentions=False,
                **block_kwargs,
            )
            if use_cache and isinstance(out, tuple) and len(out) >= 2:
                x = out[0]
                presents.append(out[1])
            else:
                x = out[0] if isinstance(out, tuple) else out

            _free_block_weights(container[i])

        if norm_w is not None:
            if arch == "llama":
                x = _rms_norm(x, norm_w, getattr(model.config, "rms_norm_eps", 1e-6))
            else:
                x = F.layer_norm(x, (x.shape[-1],), norm_w, norm_b)
        logits = F.linear(x, lm_head_w)

        return logits

    finally:
        pass  # No restore — SVD shard doesn't have full weights


@torch.no_grad()
def svd_compare_memory(model: nn.Module, tokenizer,
                        svd_dir: str, texts,
                        max_length: int = 128,
                        rank: int = 64,
                        memmap_embedding: bool = False) -> Dict[str, Any]:
    """Compare full model vs SVD-streaming vs regular streaming.

    Measures NLL, peak RSS, compression savings, and SVD approximation
    quality across all three modes.

    Returns comprehensive report dict.
    """
    device = _get_device(model)
    n_layers = _n_layers(model)
    d = _d_model(model)
    result: Dict[str, Any] = {}

    free_memory_trim()
    gc.collect()
    rss_before = measure_memory_rss()

    t0 = time.perf_counter()
    nll_full = _eval_nll_full(model, tokenizer, texts, max_length)
    t_full = time.perf_counter() - t0
    rss_full = measure_memory_rss()
    result["full"] = {
        "nll": nll_full,
        "time_s": t_full,
        "rss_mb": rss_full / 1024 / 1024,
        "rss_delta_mb": (rss_full - rss_before) / 1024 / 1024,
    }

    free_memory_trim()
    _free_all_block_weights(model)
    free_memory_trim()
    rss_bare = measure_memory_rss()

    rss_peak = rss_bare
    def _track():
        nonlocal rss_peak
        rss_peak = max(rss_peak, measure_memory_rss())

    t0 = time.perf_counter()
    nll_svd = _eval_nll_streaming_svd(
        model, tokenizer, texts, svd_dir, max_length, rank,
        memmap_embedding, _track_fn=_track,
    )
    t_svd = time.perf_counter() - t0

    result["svd_streaming"] = {
        "nll": nll_svd,
        "nll_delta": nll_svd - nll_full,
        "time_s": t_svd,
        "time_ratio": t_svd / max(t_full, 1e-9),
        "rss_bare_mb": rss_bare / 1024 / 1024,
        "rss_peak_mb": rss_peak / 1024 / 1024,
        "rss_peak_delta_mb": (rss_peak - rss_bare) / 1024 / 1024,
    }

    result["model"] = {
        "n_layers": n_layers,
        "d_model": d,
        "total_params": sum(p.numel() for p in model.parameters()),
    }

    svd_info = _compute_svd_shard_sizes(svd_dir)
    result["svd_compression"] = svd_info

    return result


def _compute_svd_shard_sizes(svd_dir: str) -> Dict[str, Any]:
    """Compute SVD compression statistics from shard directory."""
    sizes: Dict[str, int] = {}
    for fname in os.listdir(svd_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(svd_dir, fname)
        w = torch.load(path, map_location="cpu", weights_only=True)
        total = sum(v.numel() * v.element_size() for v in w.values())
        sizes[fname.replace(".pt", "")] = total
        del w

    shared_bytes = sizes.get("shared", 0)
    layer_bytes = [v for k, v in sizes.items() if k.startswith("layer_")]
    svd_total = shared_bytes + sum(layer_bytes)

    # Estimate original size from SVD factors
    orig_est = 0
    for fname in os.listdir(svd_dir):
        if not fname.startswith("layer_") or not fname.endswith(".pt"):
            continue
        path = os.path.join(svd_dir, fname)
        w = torch.load(path, map_location="cpu", weights_only=True)
        for k, v in w.items():
            if k.endswith(".U_eff"):
                base = k[:-6]
                vh_key = f"{base}.Vh"
                if vh_key in w:
                    vh = w[vh_key]
                    m, r = v.shape
                    r2, n = vh.shape
                    orig_est += m * n * v.element_size()
        del w

    total_orig_bytes = orig_est + shared_bytes
    total_svd_bytes = svd_total

    return {
        "shared_mb": shared_bytes / 1024 / 1024,
        "per_layer_svd_mb": [b / 1024 / 1024 for b in layer_bytes],
        "total_svd_mb": total_svd_bytes / 1024 / 1024,
        "estimated_original_mb": total_orig_bytes / 1024 / 1024,
        "compression_ratio": total_orig_bytes / max(total_svd_bytes, 1),
    }


def _eval_nll_streaming_svd(model, tokenizer, texts, svd_dir,
                             max_length, rank, memmap_embedding,
                             _track_fn=None) -> float:
    """Evaluate NLL using SVD-streamed weights."""
    device = _get_device(model)
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        if _track_fn:
            _track_fn()
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length)
        input_ids = enc.input_ids.to(device)
        logits = svd_stream_forward(
            model, input_ids, svd_dir, rank=rank,
            memmap_embedding=memmap_embedding, restore=False,
        )
        if _track_fn:
            _track_fn()
        if logits is None:
            continue
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        n_tok = shift_labels.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


def svd_calibrate_deltas(model: nn.Module, svd_dir: str,
                          tokenizer, texts,
                          delta_rank: int = 8,
                          steps: int = 20,
                          lr: float = 1e-4,
                          max_length: int = 128) -> Dict[str, Any]:
    """Fine-tune SVD-truncated weights on calibration data, re-extract factors.

    The core loop:
        1. Reconstruct weights from SVD factors (U_eff @ Vh)
        2. Train ALL block weights to minimize NLL on calibration data
        3. Re-SVD the trained weights back to rank-r → updated factors saved to disk

    After training, the SVD factors in svd_dir are the best rank-r approximation
    of the fine-tuned weights — task-optimal, not Frobenius-optimal.

    This is the minimal viable generative law: the factors are "trained" by the
    data distribution, producing better approximations for the actual task.
    """
    device = _get_device(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    n_layers = len(container)

    # Phase 1: Reconstruct all blocks from SVD factors
    for i in range(n_layers):
        _load_block_from_svd(container[i], svd_dir, i, device)

    # Phase 2: Train all 2D weight params on calibration data
    trainable = []
    for block in container:
        for p in block.parameters():
            if p.dim() == 2:
                p.requires_grad_(True)
                trainable.append(p)

    optimizer = torch.optim.Adam(trainable, lr=lr)
    losses = []

    for step in range(steps):
        total_loss = 0.0
        n = 0
        for text in texts:
            enc = tokenizer(text, return_tensors='pt', truncation=True,
                            max_length=max_length)
            input_ids = enc.input_ids.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = model(input_ids, labels=input_ids)
                loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        losses.append(avg_loss)

    final_loss = losses[-1] if losses else float('inf')

    # Phase 3: Re-SVD trained weights → update factors on disk
    state = model.state_dict()

    for i in range(n_layers):
        pat = _layer_pattern(i, arch)
        path = os.path.join(svd_dir, f"layer_{i}.pt")
        svd_factors = torch.load(path, map_location="cpu", weights_only=True)

        arch_prefix = _arch_layer_prefix(i, arch)
        prefix_len = len(arch_prefix)

        for key, tensor in state.items():
            if not pat.search(key) or tensor.dim() < 2:
                continue

            relative = key[prefix_len:] if prefix_len and key.startswith(arch_prefix) else key
            base = relative[:-7] if relative.endswith('.weight') and len(relative) > 7 else relative
            u_key = f"{base}.U_eff"

            if u_key not in svd_factors:
                continue

            orig_rank = svd_factors[u_key].shape[1]
            W = tensor.float().cpu()
            m, n = W.shape
            r = min(orig_rank, m, n)

            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            U_eff = U[:, :r] * S[:r].unsqueeze(0)
            Vh_trunc = Vh[:r, :].contiguous()

            svd_factors[f"{base}.U_eff"] = U_eff
            svd_factors[f"{base}.Vh"] = Vh_trunc

        torch.save(svd_factors, path)

    return {
        "steps": steps,
        "final_loss": final_loss,
        "losses": losses,
        "n_params_trained": len(trainable),
    }
    """Evaluate NLL using SVD-streamed weights."""
    device = _get_device(model)
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        if _track_fn:
            _track_fn()
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length)
        input_ids = enc.input_ids.to(device)
        logits = svd_stream_forward(
            model, input_ids, svd_dir, rank=rank,
            memmap_embedding=memmap_embedding, restore=False,
        )
        if _track_fn:
            _track_fn()
        if logits is None:
            continue
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        n_tok = shift_labels.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)
