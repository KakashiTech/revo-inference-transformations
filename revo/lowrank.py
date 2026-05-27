from __future__ import annotations
from revo._logging import get_logger

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import orient_weight, set_by_name, skip_tied_weights

def _orient_weight_bias(module: Any, out_hint: int | None = None, in_hint: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (W_oriented, b) where W_oriented is [out, in] matching bias length or hints.

    Preference order:
    1) If bias exists, choose orientation whose first dim equals bias length.
    2) Else, if hints provided, choose orientation matching (out_hint,in_hint) or (in_hint,out_hint).
    3) Fallback: keep as-is.
    """
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    if isinstance(W, torch.Tensor) and W.dim() == 2:
        if b_det is not None:
            if W.shape[0] == b_det.numel():
                return W.detach(), b_det
            if W.shape[1] == b_det.numel():
                return W.detach().t(), b_det
        if out_hint is not None and in_hint is not None:
            if W.shape == (out_hint, in_hint):
                return W.detach(), b_det
            if W.shape == (in_hint, out_hint):
                return W.detach().t(), b_det
        return W.detach(), b_det
    raise AssertionError("module.weight must be a 2D tensor")


class LowRankLinear(nn.Module):
    """Two-stage low-rank approximation of a Linear layer: W ≈ U @ V.

    Forward: y = B(A(x)) where A: in->r (weight=V), B: r->out (weight=U).
    Optionally includes per-output calibration (alpha,beta).
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.A = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.B = nn.Linear(self.rank, self.out_features, bias=bias, **factory_kwargs)
        # Per-output calibration (initialized to identity/no-op)
        self.register_buffer("alpha", torch.ones(self.out_features, **{k: v for k, v in factory_kwargs.items() if k == "device"}), persistent=False)
        self.register_buffer("beta", torch.zeros(self.out_features, **{k: v for k, v in factory_kwargs.items() if k == "device"}), persistent=False)

    @torch.no_grad()
    def set_factors(self, U: torch.Tensor, V: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        """Set factor matrices: U (out, rank), V (rank, in)."""
        assert U.shape == (self.out_features, self.rank)
        assert V.shape == (self.rank, self.in_features)
        self.B.weight.copy_(U)
        self.A.weight.copy_(V)
        if bias is not None and self.B.bias is not None:
            self.B.bias.copy_(bias)

    @torch.no_grad()
    def calibrate_from_samples(self, original: Any, n_samples: int = 256, seed: int = 0, inputs: torch.Tensor | None = None) -> None:
        """Fit per-output alpha/beta such that y_true ≈ alpha ⊙ y_hat + beta.
        If `inputs` is provided, use them as calibration inputs; otherwise sample Gaussian.
        """
        # Expect original to expose .weight (2D) and optional .bias
        assert hasattr(original, "weight"), "original must have .weight"
        if isinstance(inputs, torch.Tensor):
            X = inputs.to(device=self.A.weight.device, dtype=self.A.weight.dtype)
            # Flatten to [N, in_features]
            if X.dim() > 2:
                X = X.view(-1, X.shape[-1])
        else:
            gen = torch.Generator(device=self.A.weight.device)
            gen.manual_seed(int(seed))
            X = torch.randn(n_samples, self.in_features, device=self.A.weight.device, dtype=self.A.weight.dtype, generator=gen)
        with torch.no_grad():
            y_true = None
            # Prefer calling the original module if possible (handles Conv1D orientation)
            try:
                y_try = original(X)
                if y_try is not None:
                    y_true = y_try
            except Exception:
                y_true = None
            if y_true is None:
                # Fallback: orient weights to [out,in] and use F.linear
                W_use, b_use = _orient_weight_bias(original, out_hint=self.out_features, in_hint=self.in_features)
                W_use = W_use.to(device=X.device, dtype=X.dtype)
                b_use = b_use.to(device=X.device, dtype=X.dtype) if isinstance(b_use, torch.Tensor) else None
                y_true = F.linear(X, W_use, b_use)
            y_hat = self.forward(X)
        # Ensure 2D shape [n_samples, out_features]
        if y_true.dim() > 2:
            y_true = y_true.view(n_samples, -1)
        # Compute per-output alpha, beta via least squares closed form
        # alpha_j = sum(yh_j * yt_j) / sum(yh_j^2); beta_j = mean(yt_j - alpha_j*yh_j)
        eps = 1e-12
        num = (y_hat * y_true).sum(dim=0)
        den = (y_hat * y_hat).sum(dim=0) + eps
        alpha = num / den
        beta = (y_true - y_hat * alpha).mean(dim=0)
        self.alpha.copy_(alpha)
        self.beta.copy_(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.A(x)
        y = self.B(z)
        # Apply calibration only if non-trivial
        if hasattr(self, "alpha") and hasattr(self, "beta"):
            if self.alpha is not None and self.beta is not None:
                y = y * self.alpha + self.beta
        return y

    @torch.no_grad()
    def fold_calibration(self) -> None:
        """Fold per-output (alpha,beta) into B's weight/bias and neutralize buffers.

        After folding, forward no longer needs to apply calibration.
        """
        if not (hasattr(self, "alpha") and hasattr(self, "beta")):
            return
        if self.alpha is None or self.beta is None:
            return
        # Ensure shapes
        alpha = self.alpha.view(-1)
        beta = self.beta.view(-1)
        assert alpha.numel() == self.out_features and beta.numel() == self.out_features
        # Fold into B: W' = diag(alpha) @ W, b' = alpha ⊙ b + beta
        W = self.B.weight
        W.mul_(alpha.view(-1, 1))
        if self.B.bias is None:
            self.B.bias = nn.Parameter(torch.zeros(self.out_features, device=W.device, dtype=W.dtype))
        self.B.bias.mul_(alpha).add_(beta.to(device=self.B.bias.device, dtype=self.B.bias.dtype))
        # Neutralize calibration
        self.alpha.fill_(1.0)
        self.beta.zero_()


@torch.no_grad()
def compress_linear_to_lowrank(module: Any, rank: int) -> Tuple[LowRankLinear, torch.Tensor, torch.Tensor]:
    """SVD-based low-rank compression of a Linear layer.

    Returns (wrapper, U, V) where U in R[out, rank], V in R[rank, in].
    """
    # Generic module with 2D weight (e.g., nn.Linear, GPT-2 Conv1D)
    assert hasattr(module, "weight") and isinstance(module.weight, torch.Tensor) and module.weight.dim() == 2
    device = module.weight.device
    dtype = module.weight.dtype
    # Orient weight so that W is [out, in]; bias length must match out
    W_oriented, b_oriented = _orient_weight_bias(module)
    W = W_oriented.to(device="cpu", dtype=torch.float32)
    # SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = int(max(1, min(rank, U.shape[1])))
    U_r = (U[:, :r] * S[:r]).to(dtype=dtype, device=device)
    V_r = Vh[:r, :].to(dtype=dtype, device=device)

    in_features = int(W.shape[1])
    out_features = int(W.shape[0])
    wrapper = LowRankLinear(in_features, out_features, r, bias=(b_oriented is not None), device=device, dtype=dtype)
    bias = b_oriented.to(device=device, dtype=dtype) if b_oriented is not None else None
    wrapper.set_factors(U_r, V_r, bias=bias)
    return wrapper, U_r, V_r


@torch.no_grad()
def replace_linear_with_lowrank(model: nn.Module, ranks: Dict[str, int], calibrate: bool = True, calibrate_samples: int = 256, seed: int = 0, fold_calibration: bool = True, non_increase_params: bool = True) -> Dict[str, Tuple[int, int]]:
    """Replace named Linear modules with LowRankLinear using given ranks.

    Returns a dict with mapping name -> (old_rank, new_rank).
    """
    report: Dict[str, Tuple[int, int]] = {}

    def set_by_name(root: nn.Module, path: str, new_mod: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            if p.isdigit():
                parent = getattr(parent, "_modules")[p]  # ModuleList/Sequential index
            else:
                parent = getattr(parent, p)
        last = parts[-1]
        if last.isdigit():
            parent._modules[last] = new_mod
        else:
            setattr(parent, last, new_mod)

    # Detect output embedding head to preserve weight tying (e.g., GPT-2 lm_head)
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
        if name in ranks and isinstance(m, nn.Linear) and m.weight.dim() == 2:
            # Skip if this module is the tied output embedding head by identity or name
            if head_module is not None and m is head_module:
                continue
            if name == "lm_head":
                continue
            # Skip if shares weight tensor with input embeddings (tied weights)
            try:
                if (input_embed_weight is not None) and (m.weight is input_embed_weight):
                    continue
            except Exception:
                continue
            r = int(ranks[name])
            # Skip if allocator decided no compression
            if r <= 0:
                continue
            if non_increase_params:
                out_f, in_f = int(m.weight.shape[0]), int(m.weight.shape[1])
                # structural cap: ensure r <= (out*in)/(out+in)
                thr = max(1, (out_f * in_f) // (out_f + in_f))
                if r > thr:
                    r = int(thr)
            new_mod, U_r, V_r = compress_linear_to_lowrank(m, r)
            if calibrate:
                new_mod.calibrate_from_samples(m, n_samples=calibrate_samples, seed=seed)
                if fold_calibration:
                    new_mod.fold_calibration()
            set_by_name(model, name, new_mod)
            report[name] = (m.weight.shape[0], r)
    return report


@torch.no_grad()
def replace_2d_modules_with_lowrank(
    model: nn.Module,
    ranks: Dict[str, int],
    calibrate: bool = True,
    calibrate_samples: int = 256,
    seed: int = 0,
    module_inputs: Dict[str, torch.Tensor] | None = None,
    fold_calibration: bool = True,
    non_increase_params: bool = True,
) -> Dict[str, Tuple[int, int]]:
    """Replace any module with a 2D weight tensor by a low-rank approximation.

    - Skips Embedding-like modules and lm_head/tied weights.
    - Uses energy-cap ranks; if r<=0 the layer is skipped.
    - Accepts generic 2D-weight modules (e.g., GPT-2 Conv1D).
    """
    report: Dict[str, Tuple[int, int]] = {}

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

    # Detect tied components
    input_embed_weight = None
    try:
        get_inp = getattr(model, "get_input_embeddings", None)
        if callable(get_inp):
            inp = get_inp()
            if inp is not None and hasattr(inp, "weight"):
                input_embed_weight = getattr(inp, "weight")
    except Exception:
        input_embed_weight = None

    for name, m in model.named_modules():
        if name == "lm_head":
            continue
        if name not in ranks:
            continue
        r = int(ranks[name])
        if r <= 0:
            continue
        # require 2D tensor weight and skip Embedding-like
        W = getattr(m, "weight", None)
        if not isinstance(W, torch.Tensor) or W.dim() != 2:
            continue
        if "embedding" in m.__class__.__name__.lower():
            continue
        try:
            if (input_embed_weight is not None) and (W is input_embed_weight):
                continue
        except Exception:
            get_logger().warning("except Exception:")
        if non_increase_params:
            out_f, in_f = int(W.shape[0]), int(W.shape[1])
            thr = max(1, (out_f * in_f) // (out_f + in_f))
            if r > thr:
                r = int(thr)
        # Build wrapper by SVD
        new_mod, U_r, V_r = compress_linear_to_lowrank(m, r)
        if calibrate:
            X_in = None
            if isinstance(module_inputs, dict) and name in module_inputs:
                X_in = module_inputs[name]
            new_mod.calibrate_from_samples(m, n_samples=calibrate_samples, seed=seed, inputs=X_in)
            if fold_calibration:
                new_mod.fold_calibration()
        set_by_name(model, name, new_mod)
        report[name] = (int(W.shape[0]), int(r))
    return report
