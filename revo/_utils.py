"""Shared utilities for REVO modules.

Consolidates duplicated functions across the codebase.
"""
from __future__ import annotations
from revo._logging import get_logger

import gc
import json
import os
import random
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def orient_weight(
    module: nn.Module,
    out_hint: Optional[int] = None,
    in_hint: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    W = module.weight.data
    b = getattr(module, "bias", None)
    b_det = b.data if b is not None else None
    if b_det is not None:
        if b_det.numel() == W.shape[0]:
            return W, b_det, W.shape[1], W.shape[0]
        if b_det.numel() == W.shape[1]:
            return W.t(), b_det, W.shape[0], W.shape[1]
    if out_hint is not None and in_hint is not None:
        if W.shape == (out_hint, in_hint):
            return W, None, in_hint, out_hint
        if W.shape == (in_hint, out_hint):
            return W.t(), None, out_hint, in_hint
    return W, None, W.shape[1], W.shape[0]


def set_by_name(root: nn.Module, path: str, new_mod: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent._modules[p]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent._modules[last] = new_mod
    else:
        setattr(parent, last, new_mod)


def entropy_from_logits(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    h = -torch.sum(p * torch.log(p.clamp(min=1e-12))).item()
    return float(h)


def avg_nll(
    model: nn.Module,
    tok: Any,
    texts: List[str],
    max_len: int,
    device: torch.device = DEVICE,
) -> float:
    total_loss, total_tokens = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n_tok = enc["input_ids"].numel()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def skip_tied_weights(model: nn.Module, module: nn.Module, weight: torch.Tensor) -> bool:
    try:
        get_out = getattr(model, "get_output_embeddings", None)
        if callable(get_out):
            head = get_out()
            if head is not None and hasattr(head, "weight") and head.weight.data_ptr() == weight.data_ptr():
                return True
        get_inp = getattr(model, "get_input_embeddings", None)
        if callable(get_inp):
            inp = get_inp()
            if inp is not None and hasattr(inp, "weight") and inp.weight.data_ptr() == weight.data_ptr():
                return True
    except Exception:
        return False
    return False


def gen_texts(n: int) -> List[str]:
    base = [
        "Explica brevemente el filtrado espectral y su impacto.",
        "Describe el uso de matrices circulantes en FFT.",
        "¿Qué aporta HoRA sobre una variedad hiperbólica?",
        "Resume el mapeo holográfico bulk→boundary→bulk.",
        "Define rango efectivo y energía espectral.",
        "¿Qué es un bus de fase natural?",
        "Explica el un-computing reversible.",
        "¿Qué es WDM y cómo paraleliza subcanales?",
        "¿Cómo funciona un gating efímero suave?",
        "Explica el caching tipo árbol de prefijos (Radix).",
    ]
    return [base[i % len(base)] + f" [#{i}]" for i in range(n)]


# ---------------------------------------------------------------------------
# Extended shared utilities for module replacement and refactoring
# ---------------------------------------------------------------------------

def load_model_tokenizer(
    model_name: str,
    device: torch.device = DEVICE,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_kwargs = {k: v for k, v in kwargs.items() if k != "torch_dtype"}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    model.to(device)
    return model, tokenizer


def encode_text(
    tokenizer: Any,
    text: str,
    max_len: int,
    device: torch.device = DEVICE,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(device) for k, v in enc.items()}


def evaluate_nll(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    max_length: int = 128,
    device: torch.device = DEVICE,
) -> float:
    total_loss, total_tokens = 0.0, 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n_tok = enc["input_ids"].numel()
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def iter_linear_modules(
    model: nn.Module,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Generator[Tuple[str, nn.Module, torch.Tensor, int, int], None, None]:
    if name_patterns is None:
        name_patterns = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",
                         "fc", "dense", "linear", "mlp"]
    for name, m in model.named_modules():
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in name_patterns):
            continue
        if not hasattr(m, "weight") or not isinstance(m.weight, torch.Tensor):
            continue
        W = m.weight
        if W.dim() != 2:
            continue
        if skip_tied_weights(model, m, W):
            continue
        in_feat, out_feat = W.shape[1], W.shape[0]
        yield name, m, W, in_feat, out_feat


def resolve_module_path(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent._modules[p]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def save_results_json(
    result: Dict[str, Any],
    default_dir: str = "quality",
    prefix: str = "run",
    name: str = "",
) -> str:
    os.makedirs(default_dir, exist_ok=True)
    label = name or f"{prefix}_{int(time.time())}"
    path = os.path.join(default_dir, f"{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def measure_memory_rss() -> int:
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def free_memory_trim() -> None:
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        get_logger().warning("except Exception:")