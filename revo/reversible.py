from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import ctypes
from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules


def _orient_weight(module: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    if not isinstance(W, torch.Tensor) or W.dim() != 2:
        raise AssertionError("module.weight must be 2D tensor")
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    W_det = W.detach()
    if b_det is not None:
        if W_det.shape[0] == b_det.numel():
            return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0])
        if W_det.shape[1] == b_det.numel():
            return W_det.t(), b_det, int(W_det.shape[0]), int(W_det.shape[1])
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0])


class ReversibleUncomputeWrap(nn.Module):
    """Apply base then un-compute a low-rank component: y' = y - base(B(A(y))).

    A: out->r, B: r->in. Initialized from top-r SVD of base weight.
    A learnable, B fixed; gated by sigmoid(gamma).
    """

    def __init__(self, base: nn.Module, rank: int = 2, device=None, dtype=None):
        super().__init__()
        self.base = base
        W_o, b_o, in_f, out_f = _orient_weight(base)
        self.in_features = int(in_f)
        self.out_features = int(out_f)
        r = int(max(1, min(rank, min(self.in_features, self.out_features))))
        self.rank = r
        factory_kwargs = {"device": device or W_o.device, "dtype": dtype or W_o.dtype}
        # Low-rank uncompute maps
        self.A = nn.Linear(self.out_features, r, bias=False, **factory_kwargs)
        self.B = nn.Linear(r, self.in_features, bias=False, **factory_kwargs)
        # Gate (init close to no effect: sigmoid(-9) ~ 1e-4)
        self.gamma = nn.Parameter(torch.tensor(-9.0, device=factory_kwargs["device"]))
        # Init from SVD: un-compute path approximates W^{-1} via V @ S^{-1} @ U^T
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W_o.to(device="cpu", dtype=torch.float32), full_matrices=False)
            S_r = S[:r].clamp(min=1e-8)
            U_r = U[:, :r].to(**factory_kwargs)      # [out, r]
            Vh_r = Vh[:r, :].to(**factory_kwargs)    # [r, in]
            # A: out -> r,  B: r -> in
            # x_hat = B(A(y)) = (Vh_r^T @ diag(1/S_r) @ U_r^T)(y)
            self.A.weight.copy_((U_r / S_r.unsqueeze(0)).t().contiguous())  # [r, out] = U_r^T * S_r^{-1}
            self.B.weight.copy_(Vh_r.t().contiguous())                     # [in, r] = Vh_r^T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute base paths without building autograd tape (we only train gamma)
        with torch.no_grad():
            y = self.base(x)
            # Low-rank path
            z = self.A(y)
            x_hat = self.B(z)
            # Un-compute through base
            y_un = self.base(x_hat)
        # Free intermediate as soon as possible
        del z, x_hat
        # Gate acts with gradient; y/y_un are treated as constants for gamma optimization
        beta = torch.sigmoid(self.gamma)
        out = y - beta * y_un
        # Drop temporaries before returning
        del y, y_un
        # Schedule adiabatic decompute right after gradients flow through this block
        try:
            if DecomputeManager.enabled and isinstance(out, torch.Tensor) and out.requires_grad:
                def _trim_hook(grad: torch.Tensor) -> torch.Tensor:
                    try:
                        DecomputeManager.force_trim()
                    except Exception:
                        pass
                    return grad
                out.register_hook(_trim_hook)
        except Exception:
            pass
        # Optional immediate decomputation trimming
        try:
            if DecomputeManager.enabled:
                DecomputeManager.maybe_trim()
        except Exception:
            pass
        return out


class DecomputeManager:
    """Global manager to trigger immediate decomputation/trim after reversible blocks.

    When enabled, each reversible block forward will attempt to release Python refs,
    run GC, and call malloc_trim(0) at a configurable interval to reduce RSS peak on CPU.
    """

    enabled: bool = False
    trim_interval: int = 0
    counter: int = 0

    @staticmethod
    def enable(trim_interval: int = 0) -> None:
        DecomputeManager.enabled = True
        DecomputeManager.trim_interval = int(max(0, trim_interval))
        DecomputeManager.counter = 0

    @staticmethod
    def disable() -> None:
        DecomputeManager.enabled = False
        DecomputeManager.counter = 0

    @staticmethod
    def maybe_trim() -> None:
        DecomputeManager.counter += 1
        do_trim = (DecomputeManager.trim_interval <= 1) or (
            DecomputeManager.trim_interval > 1 and (DecomputeManager.counter % DecomputeManager.trim_interval == 0)
        )
        if not do_trim:
            return
        try:
            gc.collect()
        except Exception:
            pass
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    @staticmethod
    def force_trim() -> None:
        """Force an immediate trim regardless of interval."""
        try:
            gc.collect()
        except Exception:
            pass
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


@torch.no_grad()
def replace_with_reversible(
    model: nn.Module,
    rank: int = 2,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, Tuple[int, int, int]]:
    pats = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    report: Dict[str, Tuple[int, int, int]] = {}

    def set_by_name(root: nn.Module, path: str, new_mod: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else getattr(parent, "_modules")[p]
        last = parts[-1]
        if last.isdigit():
            parent._modules[last] = new_mod
        else:
            setattr(parent, last, new_mod)

    # Skip tied weights
    head_module = None
    input_embed_weight = None
    try:
        get_head = getattr(model, "get_output_embeddings", None)
        if callable(get_head):
            head_module = get_head()
    except Exception:
        head_module = None
    try:
        get_inp = getattr(model, "get_input_embeddings", None)
        if callable(get_inp):
            inp = get_inp()
            if inp is not None and hasattr(inp, "weight"):
                input_embed_weight = getattr(inp, "weight")
    except Exception:
        input_embed_weight = None

    for name, m in model.named_modules():
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in pats):
            continue
        if not hasattr(m, "weight") or not isinstance(getattr(m, "weight"), torch.Tensor) or getattr(m, "weight").dim() != 2:
            continue
        try:
            if (input_embed_weight is not None) and (m.weight is input_embed_weight):
                continue
        except Exception:
            pass
        wrapper = ReversibleUncomputeWrap(m, rank=rank, device=m.weight.device, dtype=m.weight.dtype)
        set_by_name(model, name, wrapper)
        report[name] = (int(wrapper.in_features), int(wrapper.out_features), int(wrapper.rank))
    return report


def calibrate_reversible(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    steps: int = 10,
    lr: float = 5e-2,
    lambda_phys: float = 1.0,
    max_length: int = 128,
) -> Dict[str, float]:
    params = []
    for m in model.modules():
        if isinstance(m, ReversibleUncomputeWrap):
            params.append(m.gamma)
    if not params:
        return {"updated": 0.0}
    for p in model.parameters():
        p.requires_grad = False
    for p in params:
        p.requires_grad = True
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            logits = out.logits
            d1 = logits[:, 1:, :] - logits[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            loss_phys = (d2 * d2).mean()
            loss_total = loss_total + lambda_phys * loss_phys
        loss_total.backward()
        opt.step()
    model.eval()
    return {"updated": float(len(params))}


def calibrate_reversible_mdl(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    steps: int = 10,
    lr: float = 5e-2,
    lambda_phys: float = 1.0,
    mdl_lambda: float = 0.01,
    max_length: int = 128,
) -> Dict[str, float]:
    """Calibrate reversible gates with MDL-in-loss.

    Optimizes only the gamma gates of reversible wrappers using a composite loss:
    NLL (teacher forcing) + lambda_phys * smoothness + mdl_lambda * L1(gamma).
    """
    gammas: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, ReversibleUncomputeWrap):
            gammas.append(m.gamma)
    if not gammas:
        return {"updated": 0.0, "mdl_lambda": float(mdl_lambda)}
    for p in model.parameters():
        p.requires_grad = False
    for g in gammas:
        g.requires_grad = True
    opt = torch.optim.Adam(gammas, lr=lr)
    model.train()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)
            loss_nll = out.loss
            # Smoothness (second diff on logits)
            logits = out.logits
            d1 = logits[:, 1:, :] - logits[:, :-1, :]
            d2 = d1[:, 1:, :] - d1[:, :-1, :]
            loss_phys = (d2 * d2).mean()
            # MDL penalty on gamma gates
            loss_mdl = 0.0
            for g in gammas:
                loss_mdl = loss_mdl + g.abs()
            loss_mdl = loss_mdl / max(1, len(gammas))
            loss = loss_nll + float(lambda_phys) * loss_phys + float(mdl_lambda) * loss_mdl
            loss_total = loss_total + loss
        loss_total.backward()
        opt.step()
    model.eval()
    return {"updated": float(len(gammas)), "mdl_lambda": float(mdl_lambda)}
