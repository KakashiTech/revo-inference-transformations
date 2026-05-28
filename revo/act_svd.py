"""Pure SVD compression + REVO reversible tail handles.

Standard SVD truncation minimizes ||W - W_r||_F, then stores the discarded tail
as low-rank factors for perfect revert via REVO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from revo._logging import get_logger
from revo._utils import seed_everything
from revo.lowrank import LowRankLinear, set_by_name

log = get_logger(__name__)


@dataclass
class SVDTailHandle:
    """Stores the discarded tail singular components for revert."""
    U_tail: torch.Tensor   # (out, tail_rank)
    S_tail: torch.Tensor   # (tail_rank,)
    V_tail: torch.Tensor   # (in, tail_rank)
    out_features: int
    in_features: int
    tail_rank: int
    was_transposed: bool = False  # True if original weight was [in, out] (Conv1D)


@torch.no_grad()
def _orient_W(weight: torch.Tensor, bias: Optional[torch.Tensor] = None, is_conv1d: bool = False) -> Tuple[torch.Tensor, bool]:
    """Return (W_oriented_as_[out,in], was_transposed).

    Conv1D stores weight as (in, out). Linear stores as (out, in).
    Use is_conv1d flag to determine orientation (not heuristic).
    """
    W = weight
    if is_conv1d:
        return W.T, True  # Conv1D: (in, out) → (out, in)
    return W, False  # Linear: already (out, in)


@torch.no_grad()
def compress_with_tail(
    module: Any,
    rank: int,
    calibrate_samples: int = 256,
    seed: int = 42,
    eps: float = 1e-10,
) -> Tuple[LowRankLinear, SVDTailHandle]:
    """Compress module weight via SVD truncation + store tail for revert.

    Returns (lowrank_wrapper, tail_handle).
    """
    device = module.weight.device
    dtype = module.weight.dtype

    W = module.weight.detach().float().cpu()
    bias = getattr(module, "bias", None)
    bias_cpu = bias.detach().float().cpu() if bias is not None else None

    is_conv1d = 'conv1d' in type(module).__name__.lower()
    W_oriented, was_t = _orient_W(W, bias=bias_cpu, is_conv1d=is_conv1d)
    out, inp = W_oriented.shape
    r = max(1, min(rank, min(out, inp)))

    # Pure SVD truncation (optimal rank-r in Frobenius norm)
    U, S, Vh = torch.linalg.svd(W_oriented, full_matrices=False)
    U_r = U[:, :r] * S[:r].unsqueeze(0)  # (out, r), absorbs S into left factor
    Vh_r = Vh[:r, :]  # (r, inp)

    W_comp_oriented = U_r @ Vh_r
    W_comp_oriented = torch.nan_to_num(W_comp_oriented, nan=0.0)

    # Tail = discarded components (in oriented frame)
    discard = W_oriented - W_comp_oriented
    discard = torch.nan_to_num(discard, nan=0.0)
    if discard.numel() > 0 and discard.norm() > 1e-12:
        U_d, S_d, Vh_d = torch.linalg.svd(discard, full_matrices=False)
        sig = S_d > (S_d[0] * 1e-4) if S_d.numel() > 0 else torch.tensor([], dtype=torch.bool)
        n_tail = sig.sum().item()
        if n_tail > 0:
            U_tail = U_d[:, :n_tail]
            S_tail = S_d[:n_tail]
            V_tail = Vh_d[:n_tail, :].T
        else:
            U_tail = torch.zeros((out, 0))
            S_tail = torch.zeros((0,))
            V_tail = torch.zeros((inp, 0))
    else:
        U_tail = torch.zeros((out, 0))
        S_tail = torch.zeros((0,))
        V_tail = torch.zeros((inp, 0))

    # Build LowRankLinear with SVD factors
    U_r_device = U_r.to(dtype=dtype, device=device)
    V_r_device = Vh_r.to(dtype=dtype, device=device)
    wrapper = LowRankLinear(inp, out, r, bias=(bias is not None), device=device, dtype=dtype)
    wrapper.set_factors(U_r_device, V_r_device, bias=bias_cpu.to(device=device, dtype=dtype) if bias_cpu is not None else None)

    handle = SVDTailHandle(
        U_tail=U_tail.clone(),
        S_tail=S_tail.clone(),
        V_tail=V_tail.clone(),
        out_features=out,
        in_features=inp,
        tail_rank=n_tail,
        was_transposed=was_t,
    )
    return wrapper, handle


def replace_with_act_svd_compression(
    model: nn.Module,
    ranks: Dict[str, int],
    calibrate: bool = False,
    calibrate_texts: Optional[List[str]] = None,
    tokenizer=None,
    max_length: int = 128,
    calibrate_samples: int = 256,
    fold_calibration: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, SVDTailHandle]:
    """Replace modules with SVD-compressed LowRankLinear + tail handles for revert.

    Pure SVD truncation (no activation correction — experiments showed it harms NLL).
    Returns dict of name -> SVDTailHandle for reverting.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    handles: Dict[str, SVDTailHandle] = {}
    input_embed_weight = _detect_tied_weights(model)

    for name, m in model.named_modules():
        if name not in ranks:
            continue
        r = int(ranks[name])
        if r <= 0:
            continue
        W = getattr(m, "weight", None)
        if W is None or W.dim() != 2:
            continue
        if "embedding" in m.__class__.__name__.lower():
            continue
        try:
            if (input_embed_weight is not None) and (W is input_embed_weight):
                continue
        except Exception:
            pass
        if name == "lm_head":
            continue

        log.info("Compressing %s rank=%d shape=%s", name, r, list(W.shape))
        wrapper, handle = compress_with_tail(m, r)
        set_by_name(model, name, wrapper)
        handles[name] = handle

    return handles


def _detect_tied_weights(model: nn.Module) -> Optional[torch.Tensor]:
    try:
        get_inp = getattr(model, "get_input_embeddings", None)
        if callable(get_inp):
            inp = get_inp()
            if inp is not None and hasattr(inp, "weight"):
                return inp.weight
    except Exception:
        pass
    return None


@torch.no_grad()
def _effective_weight(lr: LowRankLinear) -> torch.Tensor:
    """Get effective weight from LowRankLinear (B.weight @ A.weight) in [out, in]."""
    return lr.B.weight @ lr.A.weight


def revert_act_svd_compression(
    model: nn.Module,
    handles: Dict[str, SVDTailHandle],
) -> None:
    """Revert model to original weights using stored tail handles."""
    for name, handle in handles.items():
        for mod_name, mod in model.named_modules():
            if mod_name == name:
                # Get effective weight (LowRankLinear uses A/B factors, not .weight)
                if isinstance(mod, LowRankLinear):
                    W_cur = _effective_weight(mod).detach().float().cpu()  # already (out, in) oriented
                else:
                    W_cur = mod.weight.detach().float().cpu()
                # W_cur is in oriented frame (out, in). Tail is also in oriented frame.
                Wc_or = W_cur
                tail = handle.U_tail @ torch.diag(handle.S_tail) @ handle.V_tail.T
                W_rest_or = Wc_or + tail  # (out, in) in oriented frame
                # Transpose back to original orientation
                if handle.was_transposed:
                    W_rest = W_rest_or.T  # (in, out) for original Conv1D
                else:
                    W_rest = W_rest_or  # (out, in) for original Linear

                parent, key = _resolve_parent(model, name)
                has_bias = (hasattr(mod, "B") and hasattr(mod.B, "bias") and mod.B.bias is not None) or \
                           (hasattr(mod, "bias") and mod.bias is not None)
                orig_mod = nn.Linear(
                    handle.in_features, handle.out_features,
                    bias=has_bias, device=W_rest.device, dtype=W_rest.dtype,
                )
                if handle.was_transposed:
                    orig_mod.weight.data = W_rest.T.to(dtype=mod.B.weight.dtype if isinstance(mod, LowRankLinear) else mod.weight.dtype, device=mod.B.weight.device if isinstance(mod, LowRankLinear) else mod.weight.device)
                else:
                    orig_mod.weight.data = W_rest.to(dtype=mod.B.weight.dtype if isinstance(mod, LowRankLinear) else mod.weight.dtype, device=mod.B.weight.device if isinstance(mod, LowRankLinear) else mod.weight.device)
                # Copy bias if present
                if has_bias:
                    if isinstance(mod, LowRankLinear) and mod.B.bias is not None:
                        orig_mod.bias.data = mod.B.bias.data.clone()
                    elif hasattr(mod, "bias") and mod.bias is not None:
                        orig_mod.bias.data = mod.bias.data.clone()
                setattr(parent, key, orig_mod)
                break


def _resolve_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent._modules[p]
    return parent, parts[-1]
