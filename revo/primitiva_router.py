"""Primitiva Router — token-conditional computational primitive selection.

Core idea: each transformer layer can compute its output using different
mathematical primitives (dense, FFT-circulant, WDM, holography, low-rank)
selected per token by a learned router. The entire system is differentiable
via Gumbel-Softmax, so the router learns which primitive's inductive bias
best suits each symbol.

Usage:
    model = PrimitiveModel(gpt2_model, enabled=[\"dense\",\"circulant\"])
    logits = model(input_ids)
    weights = model.collect_router_weights()
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import set_by_name

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _nearest_circulant_first_column(W: torch.Tensor) -> torch.Tensor:
    assert W.dim() == 2 and W.shape[0] == W.shape[1]
    n = W.shape[0]
    idx = torch.arange(n, device=W.device)
    rows = (idx[:, None] - idx[None, :]) % n
    C = W.gather(0, rows)
    return C[:, 0].contiguous()


# ─── Primitive base ───────────────────────────────────────────────────────────

class Primitive(nn.Module):
    """Base class for one computational primitive variant."""
    name: str = "base"

    def describe(self) -> Dict[str, object]:
        return {"name": self.name, "params": sum(p.numel() for p in self.parameters())}


# ─── Concrete primitives ──────────────────────────────────────────────────────

class DensePrimitive(Primitive):
    """Original dense Linear: y = xW^T + b."""
    name = "dense"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer("W", weight.detach().clone())
        if bias is not None:
            self.register_buffer("b", bias.detach().clone())
        else:
            self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.W, self.b)


class CirculantPrimitive(Primitive):
    """Circulant matrix via FFT: y = ifft(fft(c) * fft(x)) + b.
    Requires in_features == out_features.
    """
    name = "circulant"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        n = weight.shape[0]
        assert weight.shape[0] == weight.shape[1], "Circulant requires square"
        self.N = n
        self.register_buffer("c", _nearest_circulant_first_column(weight))
        if bias is not None:
            self.register_buffer("b", bias.detach().clone())
        else:
            self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_freq = torch.fft.rfft(self.c)
        x_freq = torch.fft.rfft(x, dim=-1)
        y = torch.fft.irfft(x_freq * c_freq, n=self.N, dim=-1)
        if self.b is not None:
            y = y + self.b
        return y


class WDMPrimitive(Primitive):
    """WDM block-circulant: split into B bands, per-band FFT circulant.
    Requires in_features == out_features and % bands == 0.
    """
    name = "wdm"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], bands: int = 2):
        super().__init__()
        n = weight.shape[0]
        assert weight.shape[0] == weight.shape[1] and n % bands == 0
        self.N = n
        self.bands = bands
        self.band_size = n // bands
        cols = []
        for bi in range(bands):
            s, e = bi * self.band_size, (bi + 1) * self.band_size
            cols.append(_nearest_circulant_first_column(weight[s:e, s:e]))
        self.register_buffer("c_cols", torch.stack(cols, dim=0))
        if bias is not None:
            self.register_buffer("b", bias.detach().clone())
        else:
            self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = self.bands, self.band_size
        xs = x.reshape(*x.shape[:-1], B, S)
        cfreq = torch.fft.rfft(self.c_cols, dim=-1)
        xfreq = torch.fft.rfft(xs, dim=-1)
        y = torch.fft.irfft(xfreq * cfreq, n=S, dim=-1)
        y = y.reshape(*x.shape[:-1], self.N)
        if self.b is not None:
            y = y + self.b
        return y


class HolographyPrimitive(Primitive):
    """Bulk→Boundary→Bulk: y = Wx + b + beta * W Q Mb Q^T x.
    Q ∈ R^{in×d} orthonormal, Mb ∈ R^{d×d} learnable boundary operator.
    """
    name = "holography"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], boundary_dim: int = 8):
        super().__init__()
        out_f, in_f = weight.shape
        self.in_features = in_f
        self.out_features = out_f
        d = min(boundary_dim, in_f)
        dev, dt = weight.device, weight.dtype
        self.register_buffer("W", weight.detach().clone())
        if bias is not None:
            self.register_buffer("b", bias.detach().clone())
        else:
            self.b = None
        Q = torch.randn(in_f, d, device=dev, dtype=dt)
        Q = Q / Q.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.register_buffer("Q", Q)
        self.Mb = nn.Parameter(torch.zeros(d, d, device=dev, dtype=dt))
        self.gamma = nn.Parameter(torch.tensor(0.0, device=dev if dev is not None else 'cpu'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.W, self.b)
        proj = self.Q @ self.Mb @ self.Q.T  # [in, in]
        x_holo = x @ proj
        holo = F.linear(x_holo, self.W, None)
        beta = torch.sigmoid(self.gamma)
        return base + beta * holo


class LowRankPrimitive(Primitive):
    """Low-rank adapter: y = xW^T + b + xA^TB^T (LoRA-style)."""
    name = "lowrank"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], rank: int = 4):
        super().__init__()
        out_f, in_f = weight.shape
        r = min(rank, in_f, out_f)
        dev, dt = weight.device, weight.dtype
        self.register_buffer("W", weight.detach().clone())
        if bias is not None:
            self.register_buffer("b", bias.detach().clone())
        else:
            self.b = None
        self.A = nn.Parameter(torch.randn(in_f, r, device=dev, dtype=dt) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, out_f, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.W, self.b)
        lora = x @ self.A @ self.B
        return base + lora


# ─── Primitive registry ───────────────────────────────────────────────────────

ALL_PRIMITIVES: Dict[str, type] = {
    "dense": DensePrimitive,
    "circulant": CirculantPrimitive,
    "wdm": WDMPrimitive,
    "holography": HolographyPrimitive,
    "lowrank": LowRankPrimitive,
}


def _build_primitive(name: str, weight: torch.Tensor, bias: Optional[torch.Tensor],
                     **kwargs) -> Primitive:
    cls = ALL_PRIMITIVES[name]
    return cls(weight, bias, **kwargs)


# ─── Router per layer ─────────────────────────────────────────────────────────

class PrimitiveRouter(nn.Module):
    """Per-layer router: d_model → logits over N primitives.

    Supports:
      - Soft selection (training, differentiable)
      - Hard Gumbel-Softmax (training, differentiable discrete)
      - Hard argmax (inference, efficient)
    """

    def __init__(self, d_model: int, n_primitives: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_primitives)

    def forward(self, x: torch.Tensor, temperature: float = 1.0,
                hard: bool = False) -> torch.Tensor:
        logits = self.linear(x)
        if hard and not self.training:
            idx = logits.argmax(dim=-1)
            w = torch.zeros_like(logits)
            w.scatter_(-1, idx.unsqueeze(-1), 1.0)
            return w
        if hard:
            return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        return F.softmax(logits / temperature, dim=-1)


# ─── PrimitiveSelector: one Linear → N primitives + router ───────────────────

class PrimitiveSelector(nn.Module):
    """Replaces one nn.Linear with a routing ensemble of computational primitives.

    On forward:
      1. Router produces mixing weights over primitives
      2. Each primitive computes its output
      3. Weighted sum = final output
    Weights are stored internally for inspection.
    """

    def __init__(self, in_features: int, out_features: int,
                 weight: torch.Tensor, bias: Optional[torch.Tensor],
                 enabled: Optional[List[str]] = None,
                 router_temperature: float = 1.0, router_hard: bool = True,
                 top_k: Optional[int] = None):
        super().__init__()
        self.enabled = enabled or list(ALL_PRIMITIVES.keys())
        self.router_temperature = router_temperature
        self.router_hard = router_hard
        self.in_features = in_features
        self.out_features = out_features
        self._top_k = top_k  # None = compute all

        W = weight
        b = bias

        self.primitives = nn.ModuleDict()
        for name in self.enabled:
            if name not in ALL_PRIMITIVES:
                continue
            if name == "circulant" and self.in_features != self.out_features:
                self.enabled = [n for n in self.enabled if n != "circulant"]
                continue
            if name == "wdm" and (self.in_features != self.out_features
                                  or self.in_features % 2 != 0):
                self.enabled = [n for n in self.enabled if n != "wdm"]
                continue
            kw: Dict[str, object] = {}
            if name == "wdm":
                kw["bands"] = 2
            elif name == "holography":
                kw["boundary_dim"] = min(8, self.in_features)
            elif name == "lowrank":
                kw["rank"] = min(4, self.in_features, self.out_features)
            self.primitives[name] = _build_primitive(name, W, b, **kw)

        self._names = list(self.primitives.keys())
        self.router = PrimitiveRouter(self.in_features, len(self._names))
        self._last_weights: Optional[torch.Tensor] = None

    def set_top_k(self, k: Optional[int]) -> None:
        """Set sparse compute budget (None = compute all)."""
        self._top_k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, D = x.shape
        assert D == self.in_features, f"Expected {self.in_features}, got {D}"
        flat = x.reshape(-1, D)  # [N, in]
        N = flat.shape[0]

        weights = self.router(flat, temperature=self.router_temperature,
                              hard=self.router_hard)  # [N, K]
        K = len(self._names)
        self._last_weights = weights.detach().reshape(*batch, K)

        top_k = min(self._top_k, K) if self._top_k is not None else K

        if top_k == K:
            # Dense mode: compute all primitives, weighted sum
            out = None
            for i, name in enumerate(self._names):
                w = weights[:, i:i+1]
                y = self.primitives[name](flat)
                if out is None:
                    out = y * w
                else:
                    out = out + y * w
            return out.reshape(*batch, -1) if out is not None else x

        # Sparse top-k: only compute selected primitives
        top_w, top_idx = torch.topk(weights, k=top_k, dim=-1)  # [N, k]
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        out = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)
        slot_arange = torch.arange(top_k, device=x.device, dtype=torch.int)
        for ki, name in enumerate(self._names):
            mask = (top_idx == ki).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            # Vectorized: find which slot(s) had this primitive for each selected token
            slot_of_ki = (top_idx[idx] == ki).int() @ slot_arange  # [M]
            w_sub = top_w[idx, slot_of_ki]
            out.index_add_(0, idx, self.primitives[name](flat[idx])
                           * w_sub.unsqueeze(-1))
        return out.reshape(*batch, -1)

    def describe(self) -> Dict[str, object]:
        return {
            "primitives": self._names,
            "in": self.in_features,
            "out": self.out_features,
        }


# ─── Model-level wrapper ─────────────────────────────────────────────────────

class PrimitiveModel(nn.Module):
    """Wraps a transformer, replacing Linear layers with PrimitiveSelectors.

    The router at each layer learns per token which mathematical primitive
    best suits the computation. The base model's forward is preserved.

    Args:
        model: transformer with nn.Linear in attn/mlp layers
        enabled: list of primitive names to enable
        name_patterns: module name patterns to replace
        skip_lm_head: don't replace lm_head
        router_temperature: softmax temperature
        router_hard: use hard (Gumbel) routing
    """

    def __init__(self, model: nn.Module, enabled: Optional[List[str]] = None,
                 name_patterns: Optional[List[str]] = None,
                 skip_lm_head: bool = True,
                 router_temperature: float = 1.0,
                 router_hard: bool = True,
                 top_k: Optional[int] = None):
        super().__init__()
        self.model = model
        self.enabled = enabled or list(ALL_PRIMITIVES.keys())
        self.name_patterns = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
        self.skip_lm_head = skip_lm_head
        self.router_temperature = router_temperature
        self.router_hard = router_hard
        self._top_k = top_k
        self._selectors: Dict[str, PrimitiveSelector] = {}
        self._replace_linears()

    def set_top_k(self, k: Optional[int]) -> None:
        """Change sparse compute budget for all selectors."""
        for sel in self._selectors.values():
            sel.set_top_k(k)

    def _replace_linears(self) -> None:
        replaced = 0
        for name, mod in list(self.model.named_modules()):
            if self.skip_lm_head and name == "lm_head":
                continue
            # Determine in/out features and extract weight/bias
            if isinstance(mod, nn.Linear):
                in_f, out_f = mod.in_features, mod.out_features
                W = mod.weight.data  # [out, in]
                b = mod.bias.data if mod.bias is not None else None
            elif HFConv1D is not None and isinstance(mod, HFConv1D):
                # Conv1D stores weight as [in, out] (opposite of Linear)
                in_f, out_f = mod.nx, mod.nf
                W = mod.weight.data.T.contiguous()  # -> [out, in]
                b = mod.bias.data if mod.bias is not None else None
            else:
                continue
            if not any(p in name for p in self.name_patterns):
                continue
            selector = PrimitiveSelector(
                in_f, out_f, W, b, enabled=self.enabled,
                router_temperature=self.router_temperature,
                router_hard=self.router_hard,
                top_k=self._top_k,
            )
            set_by_name(self.model, name, selector)
            self._selectors[name] = selector
            replaced += 1

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def collect_router_weights(self) -> Dict[str, torch.Tensor]:
        """Return last router weights per layer as {layer_name: [B, N]}."""
        return {name: sel._last_weights.clone()
                for name, sel in self._selectors.items()
                if sel._last_weights is not None}

    def describe(self) -> Dict[str, object]:
        return {
            "primitives": self.enabled,
            "replaced_layers": len(self._selectors),
            "router_hard": self.router_hard,
            "base_model": type(self.model).__name__,
            "per_layer": {n: s.describe() for n, s in self._selectors.items()},
        }
