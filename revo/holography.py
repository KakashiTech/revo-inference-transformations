from __future__ import annotations
from revo._logging import get_logger

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import set_by_name, skip_tied_weights


def _orient_weight_bias(module: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    """Return (W_oriented, b, in_features, out_features) with W_oriented [out,in]."""
    assert hasattr(module, "weight"), "module must have .weight"
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
    # Fallback assume already [out,in]
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0])


def _random_orthonormal(in_features: int, boundary_dim: int, device=None, dtype=None) -> torch.Tensor:
    x = torch.randn(in_features, boundary_dim, device=device, dtype=dtype)
    # QR decomposition for orthonormal columns
    q, r = torch.linalg.qr(x, mode="reduced")
    # Ensure deterministic sign
    d = torch.sign(torch.diag(r))
    q = q * d
    return q  # [in, d]


class HoloBoundaryAdapter(nn.Module):
    """Bulk→Boundary→Bulk adapter around a base Linear.

    y = W x + b + beta * W (Q M_b Q^T x)
    where Q ∈ R^{in×d} is column-orthonormal, M_b ∈ R^{d×d} is boundary operator,
    and beta = sigmoid(gamma) is a learned scalar gate calibrated by a PINN-like loss.
    """

    def __init__(self, base: nn.Module, boundary_dim: int, alpha: float = 1.0, device=None, dtype=None):
        super().__init__()
        # Infer oriented weight/bias to behave like Linear
        W_o, b_o, in_f, out_f = _orient_weight_bias(base)
        self.in_features = int(in_f)
        self.out_features = int(out_f)
        self.boundary_dim = int(max(1, min(boundary_dim, self.in_features)))
        self.alpha = float(alpha)
        factory_kwargs = {"device": device or W_o.device, "dtype": dtype or W_o.dtype}
        # Freeze base weights as buffers
        self.register_buffer("W", W_o.to(**factory_kwargs), persistent=False)
        self.register_buffer("b", (b_o.to(**factory_kwargs) if b_o is not None else None), persistent=False)
        # Boundary mapping
        Q = _random_orthonormal(self.in_features, self.boundary_dim, device=factory_kwargs["device"], dtype=factory_kwargs["dtype"])
        self.register_buffer("Q", Q, persistent=False)  # [in, d]
        self.Mb = nn.Parameter(torch.zeros(self.boundary_dim, self.boundary_dim, **factory_kwargs))
        # Gating (calibrated)
        self.gamma = nn.Parameter(torch.tensor(0.0, **{k: v for k, v in factory_kwargs.items() if k != "dtype"}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output
        y_base = F.linear(x, self.W, self.b)
        # Boundary perturbation projected back to input space
        xb = x @ self.Q  # [..., d]
        db = xb @ self.Mb  # [..., d]
        dx = db @ self.Q.t()  # [..., in]
        dy = F.linear(dx, self.W, None)  # [..., out]
        beta = torch.sigmoid(self.gamma) * self.alpha
        return y_base + beta * dy


@torch.no_grad()
def replace_with_holography(
    model: nn.Module,
    boundary_dim: int,
    alpha: float = 1.0,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, Tuple[int, int, int]]:
    patterns = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    report: Dict[str, Tuple[int, int, int]] = {}

    # Detect tied embeddings to skip
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
        if not any(pat in name for pat in patterns):
            continue
        if not hasattr(m, "weight") or not isinstance(getattr(m, "weight"), torch.Tensor) or getattr(m, "weight").dim() != 2:
            continue
        try:
            if (input_embed_weight is not None) and (m.weight is input_embed_weight):
                continue
        except Exception:
            get_logger().warning("except Exception:")
        adapter = HoloBoundaryAdapter(m, boundary_dim=boundary_dim, alpha=alpha, device=m.weight.device, dtype=m.weight.dtype)
        set_by_name(model, name, adapter)
        # Infer in/out for report via helper
        W_o, b_o, in_f, out_f = _orient_weight_bias(m)
        report[name] = (int(in_f), int(out_f), int(adapter.boundary_dim))
    return report


def _second_diff_time(logits: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V]
    d1 = logits[:, 1:, :] - logits[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return d2


def _collect_holo_gammas(model: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, HoloBoundaryAdapter):
            params.append(m.gamma)
    return params


def calibrate_holo_pinn(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    steps: int = 10,
    lr: float = 5e-2,
    lambda_phys: float = 1.0,
    max_length: int = 128,
) -> Dict[str, float]:
    # Freeze all params except gating scalars
    for p in model.parameters():
        p.requires_grad = False
    gammas = _collect_holo_gammas(model)
    for g in gammas:
        g.requires_grad = True
    if not gammas:
        return {"updated": 0.0}
    opt = torch.optim.Adam(gammas, lr=lr)
    model.train()
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            logits = out.logits  # [1, T, V]
            d2 = _second_diff_time(logits)
            loss_phys = (d2 * d2).mean()
            loss_total = loss_total + lambda_phys * loss_phys
        loss_total.backward()
        opt.step()
    model.eval()
    return {"updated": float(len(gammas))}
