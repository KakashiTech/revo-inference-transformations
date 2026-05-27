from __future__ import annotations

from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules

def _orient_weight_bias(module: nn.Module):
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    if not isinstance(W, torch.Tensor) or W.dim() != 2:
        raise AssertionError("module.weight must be 2D tensor")
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    W_det = W.detach()
    # If bias matches first dim -> already [out, in]
    if b_det is not None:
        if W_det.shape[0] == b_det.numel():
            return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False
        if W_det.shape[1] == b_det.numel():
            return W_det.t(), b_det, int(W_det.shape[0]), int(W_det.shape[1]), True
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False


class HolomorphicFourierLinearLike(nn.Module):
    """
    Linear layer parameterized by a small number of Fourier cosine modes.

    y = sum_{k=1..r} c_k * <x, cos_in(k)> * cos_out(k) + bias

    - c_k are learnable real coefficients (rank r)
    - cos_in(k)[i] = cos(2π k i / in_features)
    - cos_out(k)[j] = cos(2π k j / out_features)

    This acts as a discrete Cauchy/contour-inspired collapse to O(r) parameters,
    generating weights via Fourier basis without storing dense matrices.
    """

    def __init__(self, base: nn.Module, rank: int = 8):
        super().__init__()
        W_o, b_o, in_f, out_f, _ = _orient_weight_bias(base)
        self.in_features = int(in_f)
        self.out_features = int(out_f)
        self.rank = int(max(1, min(rank, min(self.in_features, self.out_features))))
        dev = getattr(base.weight, "device", torch.device("cpu"))
        dt = getattr(base.weight, "dtype", torch.float32)
        # Real coefficients (initialized via projection of W onto cosine bases)
        self.coeffs = nn.Parameter(torch.zeros(self.rank, device=dev, dtype=torch.float32))
        # Small per-dimension amplitude shapers (low overhead, improve fit)
        self.a_in = nn.Parameter(torch.ones(self.in_features, device=dev, dtype=torch.float32))
        self.a_out = nn.Parameter(torch.ones(self.out_features, device=dev, dtype=torch.float32))
        # Optional bias copied from base
        if isinstance(b_o, torch.Tensor):
            self.bias = nn.Parameter(b_o.to(device=dev, dtype=dt), requires_grad=False)
        else:
            self.bias = None
        # Initialize coefficients to approximate original weight
        try:
            with torch.no_grad():
                r = self.rank
                in_f = self.in_features
                out_f = self.out_features
                two_pi = 2.0 * math.pi
                # build bases on CPU float32 for numerical stability
                k = torch.arange(1, r + 1, device="cpu", dtype=torch.float32)  # [r]
                i = torch.arange(in_f, device="cpu", dtype=torch.float32)  # [in]
                j = torch.arange(out_f, device="cpu", dtype=torch.float32)  # [out]
                cos_in = torch.cos(two_pi * (k[:, None] * (i[None, :] / float(in_f))))  # [r, in]
                cos_out = torch.cos(two_pi * (k[:, None] * (j[None, :] / float(out_f))))  # [r, out]
                Wcpu = W_o.to(device="cpu", dtype=torch.float32)
                coeffs = torch.zeros(r, dtype=torch.float32)
                for idx in range(r):
                    basis_k = cos_out[idx][:, None] * cos_in[idx][None, :]  # [out, in]
                    num = (Wcpu * basis_k).sum()
                    den = (basis_k * basis_k).sum() + 1e-9
                    coeffs[idx] = (num / den)
                self.coeffs.copy_(coeffs.to(device=dev, dtype=torch.float32))
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.view(-1, orig_shape[-1])  # [N, in]
        N = x2.shape[0]
        in_f = self.in_features
        out_f = self.out_features
        r = self.rank
        device = x2.device
        dtype = x2.dtype
        # Build cosine bases on the fly (no storage)
        k = torch.arange(1, r + 1, device=device, dtype=torch.float32)  # [r]
        i = torch.arange(in_f, device=device, dtype=torch.float32)  # [in]
        j = torch.arange(out_f, device=device, dtype=torch.float32)  # [out]
        two_pi = 2.0 * math.pi
        cos_in = torch.cos(two_pi * (k[:, None] * (i[None, :] / float(in_f))))  # [r, in]
        cos_out = torch.cos(two_pi * (k[:, None] * (j[None, :] / float(out_f))))  # [r, out]
        # apply amplitude shaping
        cos_in = cos_in * self.a_in.to(dtype=cos_in.dtype, device=cos_in.device)[None, :]
        cos_out = cos_out * self.a_out.to(dtype=cos_out.dtype, device=cos_out.device)[None, :]
        # Projections
        proj = x2.to(torch.float32) @ cos_in.t()  # [N, r]
        # Weighted combination into output space
        y2 = (proj * self.coeffs[None, :]) @ cos_out  # [N, out]
        y2 = y2.to(dtype)
        if self.bias is not None:
            y2 = y2 + self.bias
        y = y2.view(*orig_shape[:-1], out_f)
        return y


def _set_by_name(root: nn.Module, path: str, new_mod: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else getattr(parent, "_modules")[p]
    last = parts[-1]
    if last.isdigit():
        parent._modules[last] = new_mod
    else:
        setattr(parent, last, new_mod)


@torch.no_grad()
def replace_mlp_with_holomorphic(
    model: nn.Module,
    rank: int = 8,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, float]:
    pats = name_patterns or ["mlp", "c_fc", "c_proj"]
    modules = 0
    orig_params = 0
    coeff_params = 0
    for name, m in model.named_modules():
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in pats):
            continue
        W = getattr(m, "weight", None)
        if not isinstance(W, torch.Tensor) or W.dim() != 2:
            continue
        # Ignore embeddings and layer norms
        if "embedding" in m.__class__.__name__.lower() or "norm" in m.__class__.__name__.lower():
            continue
        try:
            W_o, b_o, in_f, out_f, _ = _orient_weight_bias(m)
            r = int(max(1, min(rank, min(out_f, in_f))))
            orig = out_f * in_f + (out_f if getattr(m, "bias", None) is not None else 0)
            coeff = r + (out_f if getattr(m, "bias", None) is not None else 0)
            wrapper = HolomorphicFourierLinearLike(m, rank=r)
            _set_by_name(model, name, wrapper)
            modules += 1
            orig_params += orig
            coeff_params += coeff
        except Exception:
            continue
    ratio = float(coeff_params) / max(1.0, float(orig_params))
    return {
        "modules": float(modules),
        "orig_params": float(orig_params),
        "coeff_params": float(coeff_params),
        "params_ratio": float(ratio),
        "rank": float(rank),
    }


@torch.enable_grad()
def calibrate_holomorphic(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    steps: int = 20,
    lr: float = 5e-2,
    mdl_lambda: float = 0.0,
    max_length: int = 128,
) -> Dict[str, float]:
    """Calibrate holomorphic parameters (coeffs, a_in, a_out) via teacher-forcing NLL.

    Only parameters from HolomorphicFourierLinearLike are optimized; all others are frozen.
    Adds an optional MDL L1 penalty on coeffs and small L2 on amplitude shapers.
    """
    params: List[nn.Parameter] = []
    coeffs: List[nn.Parameter] = []
    amps: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, HolomorphicFourierLinearLike):
            coeffs.append(m.coeffs)
            amps.append(m.a_in)
            amps.append(m.a_out)
    params = coeffs + amps
    if not params:
        return {"updated": 0.0, "mdl_lambda": float(mdl_lambda)}
    for p in model.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True
    opt = torch.optim.Adam(params, lr=float(lr))
    model.train()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), dtype=torch.float32)
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)
            loss_total = loss_total + out.loss.to(dtype=torch.float32)
        # MDL penalties
        if mdl_lambda != 0.0:
            l1 = torch.zeros((), dtype=torch.float32)
            for c in coeffs:
                l1 = l1 + c.abs().mean()
            l2 = torch.zeros((), dtype=torch.float32)
            for a in amps:
                l2 = l2 + (a * a).mean()
            loss_total = loss_total + float(mdl_lambda) * (l1 + 1e-3 * l2)
        loss_total.backward()
        opt.step()
    model.eval()
    return {"updated": float(len(params)), "mdl_lambda": float(mdl_lambda)}
