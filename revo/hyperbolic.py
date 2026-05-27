from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _c_clamp(c: float) -> float:
    return float(max(c, 1e-8))


def project_to_ball(x: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
    c = _c_clamp(c)
    if c == 0.0:
        return x
    sqrt_c = c ** 0.5
    max_norm = (1.0 / sqrt_c) - eps
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    cond = (x_norm >= max_norm)
    scale = max_norm / x_norm
    x_proj = torch.where(cond, x * scale, x)
    return x_proj


def expmap0(v: torch.Tensor, c: float) -> torch.Tensor:
    c = _c_clamp(c)
    if c == 0.0:
        return v
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    coef = torch.tanh(sqrt_c * v_norm / 2.0) / (sqrt_c * v_norm)
    x = coef * v
    return project_to_ball(x, c)


def logmap0(x: torch.Tensor, c: float) -> torch.Tensor:
    c = _c_clamp(c)
    if c == 0.0:
        return x
    x = project_to_ball(x, c)
    sqrt_c = c ** 0.5
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    coef = (2.0 / sqrt_c) * torch.atanh(sqrt_c * x_norm) / x_norm
    return coef * x


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    c = _c_clamp(c)
    if c == 0.0:
        return x + y
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    cx2 = c * x2
    cy2 = c * y2
    cxy = c * xy
    num = (1 + 2 * cxy + cy2) * x + (1 - cx2) * y
    denom = 1 + 2 * cxy + cx2 * cy2
    z = num / denom.clamp_min(1e-12)
    return project_to_ball(z, c)


def mobius_matvec(M: torch.Tensor, x: torch.Tensor, c: float) -> torch.Tensor:
    """Hyperbolic matrix-vector via tangent transform: exp0(M @ log0(x)).
    M: [out, in], x: [..., in]
    Returns: [..., out]
    """
    c = _c_clamp(c)
    if c == 0.0:
        return F.linear(x, M, bias=None)
    x_tan = logmap0(x, c)
    z = F.linear(x_tan, M, bias=None)
    y = expmap0(z, c)
    return y
