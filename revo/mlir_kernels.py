"""MLIR-style kernel optimizations for REVO.

Provides torch.compile-based fused kernels, operator fusion for FFT/circulant
paths, and memory-efficient attention/matmul primitives.

All CPU-only, pure PyTorch.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


def compile_fn(fn: Callable, backend: str = "inductor", mode: str = "max-autotune") -> Callable:
    """Try torch.compile on a callable; fallback to original if unavailable."""
    compile = getattr(torch, "compile", None)
    if compile is None:
        return fn
    try:
        return compile(fn, backend=backend, mode=mode)
    except Exception:
        return fn


def compile_module(model: torch.nn.Module, backend: str = "inductor") -> torch.nn.Module:
    """Compile an entire nn.Module with torch.compile."""
    compile = getattr(torch, "compile", None)
    if compile is None:
        return model
    try:
        return compile(model, backend=backend, mode="max-autotune")
    except Exception:
        return model


@torch.jit.script_if_tracing
def fused_circulant_forward(x: torch.Tensor, c: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fused FFT circulant multiply: y = ifft(fft(c) * fft(x)) + bias.

    Args:
        x: [..., N] input
        c: [N] circulant first column
        bias: optional [N] bias
    Returns:
        [..., N] output
    """
    N = x.size(-1)
    c_freq = torch.fft.rfft(c)
    x_freq = torch.fft.rfft(x, dim=-1)
    y = torch.fft.irfft(x_freq * c_freq, n=N, dim=-1)
    if bias is not None:
        y = y + bias
    return y


@torch.jit.script_if_tracing
def fused_wdm_forward(x: torch.Tensor, c_cols: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Fused WDM (banded circulant) forward.

    Args:
        x: [..., N] input
        c_cols: [B, S] per-band circulant first columns
        bias: optional [N] bias
    Returns:
        [..., N] output
    """
    B, S = c_cols.shape
    N = B * S
    xs = x.reshape(*x.shape[:-1], B, S)
    cfreq = torch.fft.rfft(c_cols, dim=-1)
    xfreq = torch.fft.rfft(xs, dim=-1)
    yfreq = xfreq * cfreq
    ys = torch.fft.irfft(yfreq, n=S, dim=-1)
    y = ys.reshape(*x.shape[:-1], N)
    if bias is not None:
        y = y + bias
    return y


@torch.jit.script_if_tracing
def fused_phase_shift(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Fused FFT phase rotation: y = irfft(rfft(x) * e^{i * theta}).

    Args:
        x: [..., N] input
        theta: [N//2+1] phase angles
    Returns:
        [..., N] phase-shifted output
    """
    N = x.size(-1)
    Xf = torch.fft.rfft(x, dim=-1)
    phase = torch.exp(1j * theta.to(Xf.device, dtype=Xf.dtype))
    Yf = Xf * phase
    return torch.fft.irfft(Yf, n=N, dim=-1)


@torch.no_grad()
def fused_spectral_prune(W: torch.Tensor, energy_keep: float = 0.9) -> torch.Tensor:
    """Fused spectral pruning with FFT2, threshold, and inverse FFT2.

    Args:
        W: [M, N] weight matrix
        energy_keep: fraction of energy to keep (0..1)
    Returns:
        [M, N] pruned matrix
    """
    device = W.device
    dtype = W.dtype
    X = W.to(dtype=torch.float32)
    F_mat = torch.fft.rfft2(X)
    P = (F_mat.real * F_mat.real + F_mat.imag * F_mat.imag).reshape(-1)
    total = float(torch.sum(P).item())
    if total <= 0.0 or P.numel() == 0:
        return torch.fft.irfft2(F_mat, s=X.shape).to(device=device, dtype=dtype)
    vals, idx = torch.sort(P, descending=True)
    cumsum = torch.cumsum(vals, dim=0)
    target = float(max(0.0, min(1.0, energy_keep))) * total
    k = int(torch.searchsorted(cumsum, torch.tensor(target)).item()) + 1
    k = max(1, min(k, P.numel()))
    mask = torch.zeros_like(P, dtype=torch.bool)
    mask[:k] = True
    mask_full = torch.zeros_like(F_mat, dtype=torch.bool).reshape(-1)
    mask_full[:] = mask
    mask_full = mask_full.reshape(F_mat.shape)
    F_pruned = torch.where(mask_full, F_mat, torch.zeros_like(F_mat))
    return torch.fft.irfft2(F_pruned, s=X.shape).to(device=device, dtype=dtype)


@torch.jit.script_if_tracing
def hyperbolic_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    c: float = 0.1,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Hyperbolic attention: attention in tangent space of Poincaré ball.

    Q, K, V are already in tangent space at 0 (Euclidean).
    Operates in tangent space where distance is approximately Euclidean.
    """
    d = Q.size(-1)
    s = float(scale or (1.0 / math.sqrt(d)))
    attn = (Q @ K.transpose(-2, -1)) * s
    attn = F.softmax(attn, dim=-1)
    return attn @ V
