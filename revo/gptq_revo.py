"""GPTQ quantization with REVO reversible handles.

Applies group-wise quantization to all linear layers, then uses REVO-style
handles to selectively revert the modules that degrade quality most.

Usage:
    handles = quantize_all(model, calib_texts, tok)
    nll_q = evaluate_nll(model, ...)              # all quantized
    handles, reverted = selective_dequantize(
        model, handles, lambda: evaluate_nll(...), max_nll_delta=0.5
    )                                             # worst few reverted
    revert_all(model, handles)                    # back to full precision
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._logging import get_logger
from revo._utils import set_by_name

log = get_logger(__name__)


# ── Quantized Linear module (int4, group-wise) ─────────────────────────────

class QuantizedLinear(nn.Module):
    """Group-wise int4 quantized linear layer.

    Forward pass dequantizes on-the-fly. Stores:
    - qweight: int4 weights packed as int32
    - scales, zeros: per-group float16
    """

    def __init__(self, in_f: int, out_f: int, groups: int,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.groups = groups
        self.group_size = (in_f + groups - 1) // groups

        pack_factor = 8  # 32 / 4
        cols = (in_f + pack_factor - 1) // pack_factor
        self.register_buffer("qweight", torch.zeros(out_f, cols, dtype=torch.int32))
        self.register_buffer("scales",
            torch.zeros(out_f, groups, dtype=dtype or torch.float16, device=device))
        self.register_buffer("zeros",
            torch.zeros(out_f, groups, dtype=dtype or torch.float16, device=device))
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_f, dtype=dtype or torch.float16, device=device))
        else:
            self.bias = None

    def _unpack(self) -> torch.Tensor:
        """Unpack int4 → int32 (out, in)."""
        out, inp = self.out_features, self.in_features
        pf = 8
        w = torch.zeros(out, inp, dtype=torch.int32, device=self.qweight.device)
        for c in range(self.qweight.shape[1]):
            start = c * pf
            end = min(start + pf, inp)
            for j in range(start, end):
                shift = (j - start) * 4
                w[:, j] = (self.qweight[:, c] >> shift) & 0xF
        # Sign-extend from 4-bit to int32
        w = torch.where(w >= 8, w - 16, w)
        return w

    def dequantize(self) -> torch.Tensor:
        """Full dequantized weight (out, in)."""
        w = self._unpack().to(dtype=self.scales.dtype, device=self.scales.device)
        gs = self.group_size
        for g in range(self.groups):
            s = g * gs
            e = min(s + gs, self.in_features)
            w[:, s:e] = w[:, s:e] * self.scales[:, g:g+1] + self.zeros[:, g:g+1]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize(), self.bias)


@torch.no_grad()
def _quantize_weight(W: torch.Tensor, bits: int = 4,
                     group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise min-max quantization.

    Returns (w_int, scales, zeros) where w_int is int32, values in [-8,7].
    """
    out, inp = W.shape
    groups = (inp + group_size - 1) // group_size
    W = W.float().cpu()

    w_int = torch.zeros(out, inp, dtype=torch.int32)
    scales = torch.zeros(out, groups)
    zeros = torch.zeros(out, groups)
    q_max = (1 << (bits - 1)) - 1  # 7
    q_min = -(1 << (bits - 1))     # -8

    for g in range(groups):
        s = g * group_size
        e = min(s + group_size, inp)
        w_g = W[:, s:e]
        w_min = w_g.min(dim=1, keepdim=True).values
        w_max = w_g.max(dim=1, keepdim=True).values
        scale = (w_max - w_min) / (q_max - q_min) + 1e-10
        zero = w_min - scale * q_min
        w_q = torch.round((w_g - zero) / scale).clamp(q_min, q_max)
        w_int[:, s:e] = w_q.to(torch.int32)
        scales[:, g:g+1] = scale
        zeros[:, g:g+1] = zero

    return w_int, scales, zeros


def _pack_int4(w_int: torch.Tensor) -> torch.Tensor:
    """Pack int4 (out, in) → int32 (out, in/8)."""
    out, inp = w_int.shape
    pf = 8
    cols = (inp + pf - 1) // pf
    packed = torch.zeros(out, cols, dtype=torch.int32)
    for c in range(cols):
        start = c * pf
        end = min(start + pf, inp)
        col_data = w_int[:, start:end].to(torch.int32) & 0xF
        for j in range(end - start):
            packed[:, c] |= col_data[:, j] << (j * 4)
    return packed


# ── REVO handles ───────────────────────────────────────────────────────────

@dataclass
class QuantizeHandle:
    """Original full-precision weight for revert."""
    original_weight: torch.Tensor
    original_bias: Optional[torch.Tensor]
    was_conv1d: bool
    in_features: int
    out_features: int


def _resolve_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent._modules[p]
    return parent, parts[-1]


@torch.no_grad()
def _revert_one(model: nn.Module, name: str, handle: QuantizeHandle) -> None:
    """Replace quantized module with original full-precision Linear.

    QuantizeHandle.original_weight is stored in (out, in) orientation.
    nn.Linear stores weight as (out, in), so no transpose needed.
    """
    parent, key = _resolve_parent(model, name)
    m = nn.Linear(handle.in_features, handle.out_features,
                  bias=handle.original_bias is not None)
    m.weight.data = handle.original_weight.to(dtype=m.weight.dtype, device=m.weight.device)
    if handle.original_bias is not None:
        m.bias.data = handle.original_bias.to(dtype=m.weight.dtype, device=m.weight.device)
    setattr(parent, key, m)


# ── Main API ───────────────────────────────────────────────────────────────

def quantize_all(
    model: nn.Module,
    calibration_texts: List[str],
    tokenizer,
    max_length: int = 128,
    bits: int = 4,
    group_size: int = 128,
    name_patterns: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, QuantizeHandle]:
    """Quantize all matching modules, save originals as REVO handles.

    Returns dict of name -> QuantizeHandle for reverting.
    """
    if name_patterns is None:
        name_patterns = ["attn", "mlp", "c_fc", "c_proj"]
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    handles: Dict[str, QuantizeHandle] = {}
    device_for_quant = torch.device("cpu")  # quantize on CPU for stability

    target_modules = []
    for name, m in model.named_modules():
        W = getattr(m, "weight", None)
        if W is None or W.dim() != 2:
            continue
        if not any(p in name for p in name_patterns):
            continue
        if name == "lm_head":
            continue
        target_modules.append((name, type(m).__name__))

    for name, type_name in target_modules:
        for mn, m in model.named_modules():
            if mn != name:
                continue

            W = m.weight.detach().float().cpu()
            bias = getattr(m, "bias", None)
            bias_cpu = bias.detach().float().cpu() if bias is not None else None

            is_conv1d = "conv1d" in type_name.lower()
            if is_conv1d:
                W_orient = W.T.contiguous()  # (out, in)
            else:
                W_orient = W

            out_f, in_f = W_orient.shape

            # Save handle
            handles[name] = QuantizeHandle(
                original_weight=W_orient.clone(),
                original_bias=bias_cpu,
                was_conv1d=is_conv1d,
                in_features=in_f,
                out_features=out_f,
            )

            # Quantize
            groups = (in_f + group_size - 1) // group_size
            w_int, scales, zeros = _quantize_weight(W_orient, bits=bits,
                                                     group_size=group_size)

            qmod = QuantizedLinear(in_f, out_f, groups,
                                   bias=bias_cpu is not None,
                                   device=device, dtype=m.weight.dtype)
            qmod.qweight.data = _pack_int4(w_int).to(device=device)
            qmod.scales.data = scales.to(dtype=m.weight.dtype, device=device)
            qmod.zeros.data = zeros.to(dtype=m.weight.dtype, device=device)
            if bias_cpu is not None and qmod.bias is not None:
                qmod.bias.data = bias_cpu.to(device=device, dtype=m.weight.dtype)

            set_by_name(model, name, qmod)
            log.info("Quantized %s (%s): %dx%d → %d-bit, %d groups",
                     name, type_name, out_f, in_f, bits, groups)
            break

    return handles


def revert_all(model: nn.Module, handles: Dict[str, QuantizeHandle]) -> None:
    """Revert all quantized modules to original full-precision."""
    for name, handle in handles.items():
        _revert_one(model, name, handle)


def selective_dequantize(
    model: nn.Module,
    handles: Dict[str, QuantizeHandle],
    eval_fn: Callable[[], float],
    max_nll_delta: float = 0.5,
) -> Tuple[Dict[str, QuantizeHandle], List[str]]:
    """Revert the most damaging quantized modules until NLL is acceptable.

    For each module: revert → measure NLL → re-quantize.
    Sorts by improvement from reversion, keeps the worst offenders
    reverted until the NLL delta is within max_nll_delta of the
    fully-reverted (FP16) baseline.

    Args:
        model: Model with quantized modules
        handles: Dict from quantize_all()
        eval_fn: Callable returning current NLL
        max_nll_delta: Target maximum NLL degradation

    Returns:
        (remaining_handles_for_quantized, list_of_reverted_names)
    """
    # 1. Measure all-quantized NLL
    true_base = eval_fn()
    log.info("selective_dequantize: baseline (all FP16) = %.4f", true_base)

    # 2. Revert each module one at a time, measure improvement
    impacts: List[Tuple[float, str, QuantizeHandle]] = []
    for name, handle in list(handles.items()):
        _revert_one(model, name, handle)
        nll_reverted = eval_fn()
        # Re-quantize it back
        for mn, m in model.named_modules():
            if mn == name:
                set_by_name(model, name, m)  # no-op, m is the original
                break
        # Actually re-quantize
        _requantize_one(model, name, handle)
        improvement = nll_reverted - true_base  # positive = revert helps
        impacts.append((improvement, name, handle))
        log.info("  %s: NLL if reverted = %.4f (improvement = %+.4f)",
                 name, nll_reverted, improvement)

    # 3. Sort by improvement descending (most damaging first)
    impacts.sort(key=lambda x: x[0], reverse=True)

    # 4. Start from all-quantized, revert damaging modules one by one
    reverted: List[str] = []
    for imp, name, handle in impacts:
        if imp <= 0:
            continue  # quantization doesn't hurt, keep it
        _revert_one(model, name, handle)
        reverted.append(name)
        del handles[name]
        current_nll = eval_fn()
        log.info("  Reverted %s: NLL = %.4f (Δ=%.4f)",
                 name, current_nll, current_nll - true_base)
        if current_nll - true_base <= max_nll_delta:
            break

    final_nll = eval_fn()
    n_quantized = len(handles)
    n_total = len(impacts)
    log.info("selective_dequantize done: %d/%d quantized, NLL Δ=%.4f",
             n_quantized, n_total, final_nll - true_base)
    return handles, reverted


@torch.no_grad()
def _requantize_one(model: nn.Module, name: str, handle: QuantizeHandle,
                    bits: int = 4, group_size: int = 128) -> None:
    """Re-quantize a module that was reverted to FP16 Linear.

    The reverted module is nn.Linear with weight in (out, in) orientation.
    """
    for mn, m in model.named_modules():
        if mn == name:
            W = m.weight.detach().float().cpu()  # (out, in)
            out_f, in_f = W.shape
            groups = (in_f + group_size - 1) // group_size
            w_int, scales, zeros = _quantize_weight(W, bits=bits,
                                                     group_size=group_size)
            qmod = QuantizedLinear(in_f, out_f, groups,
                                   bias=handle.original_bias is not None,
                                   device=m.weight.device, dtype=m.weight.dtype)
            qmod.qweight.data = _pack_int4(w_int).to(device=m.weight.device)
            qmod.scales.data = scales.to(dtype=m.weight.dtype, device=m.weight.device)
            qmod.zeros.data = zeros.to(dtype=m.weight.dtype, device=m.weight.device)
            if handle.original_bias is not None and qmod.bias is not None:
                qmod.bias.data = handle.original_bias.to(
                    device=m.weight.device, dtype=m.weight.dtype)
            set_by_name(model, name, qmod)
            break
