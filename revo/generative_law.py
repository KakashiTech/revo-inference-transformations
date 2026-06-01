"""Phase XI — F generativa: la ley que genera su propia álgebra.

Cada token recibe un código discreto de 32 bits que determina QUÉ operación
matemática ejecutar y con QUÉ parámetros. El código es generado por una
ley compacta (MLP) y entrenado via Straight-Through Estimator.

Soporta dos modos:
  - 'pure':  y = gen(x)   — la ley ES el cómputo (revolucionario)
  - 'residual': y = orig(x) + eps * gen(x)   — preserva pesos preentrenados

Y top-k sparse: solo ejecuta las k primitivas más probables por token.
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


# ─── Primitive Bank ────────────────────────────────────────────────────

class PrimitiveBank(nn.ModuleDict):
    """Holds all available primitives, each parameterizable by a code."""

    PRIMITIVE_NAMES = [
        "dense",      # 0: full matmul
        "circulant",  # 1: FFT O(n log n)
        "wdm",        # 2: banded FFT
        "holography", # 3: bulk→boundary→bulk
        "lowrank",    # 4: LoRA-style
    ]

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor],
                 enabled: Optional[List[str]] = None):
        super().__init__()
        names = enabled or self.PRIMITIVE_NAMES
        out_f, in_f = weight.shape

        for name in names:
            ok = True
            if name == "dense":
                mod = _DensePrimitive(weight, bias)
            elif name == "circulant":
                ok = (in_f == out_f)
                mod = _CirculantPrimitive(weight, bias) if ok else None
            elif name == "wdm":
                ok = (in_f == out_f and in_f % 2 == 0)
                mod = _WDMPrimitive(weight, bias, bands=2) if ok else None
            elif name == "holography":
                mod = _HolographyPrimitive(weight, bias)
            elif name == "lowrank":
                mod = _LowRankPrimitive(weight, bias)
            else:
                continue
            if not ok: continue
            self[name] = mod
        self._name_list = list(self.keys())
        self._name_to_idx = {n: i for i, n in enumerate(self._name_list)}


class _DensePrimitive(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.register_buffer("W", weight.detach().clone())
        self.b = bias.detach().clone() if bias is not None else None
    def forward(self, x: torch.Tensor, param: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = param.unsqueeze(-1) if param is not None else 1.0
        return F.linear(x, self.W, self.b) * s


class _CirculantPrimitive(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        n = weight.shape[0]; self.N = n
        idx = torch.arange(n)
        rows = (idx[:, None] - idx[None, :]) % n
        C = weight.gather(0, rows)[:, 0].contiguous()
        self.register_buffer("c_freq", torch.fft.rfft(C))
        self.b = bias.detach().clone() if bias is not None else None
    def forward(self, x: torch.Tensor, param: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * self.c_freq, n=self.N, dim=-1)
        if self.b is not None: y = y + self.b
        return y * (param.unsqueeze(-1) if param is not None else 1.0)


class _WDMPrimitive(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], bands: int = 2):
        super().__init__()
        n = weight.shape[0]; self.N = n; self.bands = bands; self.band_size = n // bands
        cols = []
        for bi in range(bands):
            s, e = bi * self.band_size, (bi + 1) * self.band_size
            idx = torch.arange(self.band_size)
            rows = (idx[:, None] - idx[None, :]) % self.band_size
            cols.append(weight[s:e, s:e].gather(0, rows)[:, 0].contiguous())
        self.register_buffer("c_cols_freq", torch.fft.rfft(torch.stack(cols, dim=0), dim=-1))
        self.b = bias.detach().clone() if bias is not None else None
    def forward(self, x: torch.Tensor, param: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S = self.bands, self.band_size
        xs = x.reshape(*x.shape[:-1], B, S)
        y = torch.fft.irfft(torch.fft.rfft(xs, dim=-1) * self.c_cols_freq, n=S, dim=-1)
        y = y.reshape(*x.shape[:-1], self.N)
        if self.b is not None: y = y + self.b
        return y * (param.unsqueeze(-1) if param is not None else 1.0)


class _HolographyPrimitive(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        out_f, in_f = weight.shape; d = min(8, in_f)
        self.register_buffer("W", weight.detach().clone())
        self.b = bias.detach().clone() if bias is not None else None
        Q = torch.randn(in_f, d, device=weight.device, dtype=weight.dtype)
        Q = Q / Q.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.register_buffer("Q", Q)
        self.Mb = nn.Parameter(torch.zeros(d, d, device=weight.device, dtype=weight.dtype))
        self.gamma = nn.Parameter(torch.tensor(0.0))
    def forward(self, x: torch.Tensor, param: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = F.linear(x, self.W, self.b)
        holo = F.linear(x @ (self.Q @ self.Mb @ self.Q.T), self.W, None)
        result = base + torch.sigmoid(self.gamma) * holo
        return result * (param.unsqueeze(-1) if param is not None else 1.0)


class _LowRankPrimitive(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        out_f, in_f = weight.shape; r = min(4, in_f, out_f)
        self.register_buffer("W", weight.detach().clone())
        self.b = bias.detach().clone() if bias is not None else None
        self.A = nn.Parameter(torch.randn(in_f, r, dtype=weight.dtype) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, out_f, dtype=weight.dtype))
    def forward(self, x: torch.Tensor, param: Optional[torch.Tensor] = None) -> torch.Tensor:
        result = F.linear(x, self.W, self.b) + x @ self.A @ self.B
        return result * (param.unsqueeze(-1) if param is not None else 1.0)


# ─── Generative Law ────────────────────────────────────────────────────

class GenerativeLaw(nn.Module):
    """Maps token embedding → 32-bit discrete code via Straight-Through Estimator."""

    def __init__(self, d_model: int, hidden: int = 64, n_bits: int = 32):
        super().__init__()
        self.n_bits = n_bits
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, n_bits),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0,
                hard: bool = True) -> torch.Tensor:
        logits = self.net(x)
        probs = torch.sigmoid(logits / temperature)
        if not hard:
            return probs
        hard_code = (probs > 0.5).float()
        if self.training:
            return hard_code.detach() + probs - probs.detach()
        return hard_code


# ─── Structure Decoder ─────────────────────────────────────────────────

class StructureDecoder(nn.Module):
    """Learned decoder: 32-bit code → logits over primitives + parameters.

    Bit layout:
      0-15:  latent code  (16 bits → learned projection to primitive logits)
      16-19: scale_param  (4 bits → 16 values in [0.25, 4.0])
      20-23: temp_param   (4 bits → 16 values in [0.25, 4.0])
      24-31: reserved
    """

    PARAM_MAP = torch.linspace(0.25, 4.0, 16)

    def __init__(self, n_primitives: int):
        super().__init__()
        self.n_primitives = n_primitives
        self.code_to_logits = nn.Linear(16, n_primitives)

    def bits_to_int(self, bits: torch.Tensor) -> torch.Tensor:
        weights = 2 ** torch.arange(bits.shape[-1] - 1, -1, -1,
                                    device=bits.device, dtype=bits.dtype)
        return (bits * weights).sum(dim=-1)

    def forward(self, code: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = code[..., 0:16]
        scale_bits = code[..., 16:20]
        temp_bits = code[..., 20:24]
        prim_logits = self.code_to_logits(latent)

        def interpolate(bits, table):
            raw = self.bits_to_int(bits).clamp(min=0.0, max=len(table) - 1)
            idx = raw.float()
            lo = idx.floor().long().clamp(min=0, max=len(table) - 1)
            hi = (lo + 1).clamp(max=len(table) - 1)
            frac = (idx - idx.floor()).clamp(0.0, 1.0)
            return table[lo] * (1 - frac) + table[hi] * frac

        dev = code.device
        param_map = self.PARAM_MAP.to(dev)
        return {
            "prim_logits": prim_logits,
            "scale": interpolate(scale_bits, param_map),
            "temp": interpolate(temp_bits, param_map),
        }

    @torch.no_grad()
    def decode_discrete(self, code: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = code[..., 0:16]
        scale_bits = code[..., 16:20]
        temp_bits = code[..., 20:24]
        logits = self.code_to_logits(latent)
        prim_idx = logits.argmax(dim=-1)
        pm = self.PARAM_MAP.to(code.device)
        sr = self.bits_to_int(scale_bits).long().clamp(0, 15)
        tr = self.bits_to_int(temp_bits).long().clamp(0, 15)
        return {"prim_idx": prim_idx, "scale": pm[sr], "temp": pm[tr]}


# ─── Generative Layer ──────────────────────────────────────────────────

class GenerativeLayer(nn.Module):
    """Replaces one nn.Linear with a generative law + dynamic primitives.

    Modes:
      'pure':     y = gen(x)  — la ley genera el cómputo directamente
      'residual': y = orig(x) + sigmoid(eps) * gen(x) — preserva original

    Top-k: si top_k > 0 en eval, solo ejecuta las k primitivas más probables.
    """

    def __init__(self, in_features: int, out_features: int,
                 weight: torch.Tensor, bias: Optional[torch.Tensor],
                 enabled: Optional[List[str]] = None,
                 law_hidden: int = 64,
                 mode: str = "pure",
                 orig_layer: Optional[nn.Module] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode

        self.bank = PrimitiveBank(weight, bias, enabled=enabled)
        self.law = GenerativeLaw(in_features, hidden=law_hidden, n_bits=32)
        self.decoder = StructureDecoder(len(self.bank._name_list))
        self._last_code: Optional[torch.Tensor] = None
        self._name_list = self.bank._name_list
        self._name_to_idx = self.bank._name_to_idx
        self.top_k: Optional[int] = None  # set at inference time

        if mode == "residual":
            assert orig_layer is not None, "residual mode needs orig_layer"
            self.orig = orig_layer
            self.eps = nn.Parameter(torch.tensor(0.01))
        else:
            self.orig = None
            self.eps = None

    def _gen_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Pure generative forward. Returns (output, decoded_dict)."""
        *batch, D = x.shape
        flat = x.reshape(-1, D)
        N = flat.shape[0]

        code = self.law(flat)
        self._last_code = code.detach().reshape(*batch, 32)

        decoded = self.decoder(code)
        prim_logits = decoded["prim_logits"]
        scale = decoded["scale"].unsqueeze(-1)

        top_k = self.top_k if (not self.training and self.top_k is not None) else None

        if top_k is not None:
            # Sparse: only compute top-k primitives per token
            top_vals, top_idx = torch.topk(prim_logits, top_k, dim=-1)
            weights = F.softmax(top_vals / 1.0, dim=-1)
            out = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)
            for ki in range(top_k):
                prim_i = top_idx[:, ki]
                w_i = weights[:, ki:ki+1]
                for pname, pidx in self._name_to_idx.items():
                    mask = (prim_i == pidx)
                    if mask.any():
                        out[mask] += w_i[mask] * self.bank[pname](flat[mask])
        else:
            # Full: weighted sum of all primitives
            weights = F.softmax(prim_logits / 1.0, dim=-1)
            out = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)
            for ki, name in enumerate(self._name_list):
                y = self.bank[name](flat)
                out = out + weights[:, ki:ki+1] * y

        return (out * scale).reshape(*batch, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "residual":
            return self.orig(x) + torch.sigmoid(self.eps) * self._gen_forward(x)
        return self._gen_forward(x)

    def set_top_k(self, k: Optional[int]) -> None:
        self.top_k = k

    def collect_codes(self) -> Optional[torch.Tensor]:
        return self._last_code

    def describe(self) -> Dict[str, object]:
        return {
            "primitives": self._name_list,
            "in": self.in_features,
            "out": self.out_features,
            "mode": self.mode,
            "top_k": self.top_k,
        }


# ─── GenerativeModel ───────────────────────────────────────────────────

class GenerativeModel(nn.Module):
    """Wraps a transformer, replacing Linear layers with GenerativeLayers.

    Args:
        model: transformer model (HF GPT-2 style)
        enabled: list of primitive names to use
        name_patterns: layer name patterns to replace
        mode: 'pure' or 'residual'
        law_hidden: hidden size of the GenerativeLaw MLP
    """

    def __init__(self, model: nn.Module,
                 enabled: Optional[List[str]] = None,
                 name_patterns: Optional[List[str]] = None,
                 skip_lm_head: bool = True,
                 law_hidden: int = 64,
                 mode: str = "pure"):
        super().__init__()
        self.model = model
        self.enabled = enabled or list(PrimitiveBank.PRIMITIVE_NAMES)
        self.name_patterns = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
        self.skip_lm_head = skip_lm_head
        self.law_hidden = law_hidden
        self.mode = mode
        self._layers: Dict[str, GenerativeLayer] = {}
        self._replace_linears()

    def _replace_linears(self) -> None:
        replaced = 0
        for name, mod in list(self.model.named_modules()):
            if self.skip_lm_head and name == "lm_head":
                continue
            if isinstance(mod, nn.Linear):
                in_f, out_f = mod.in_features, mod.out_features
                W = mod.weight.data
                b = mod.bias.data if mod.bias is not None else None
            elif HFConv1D is not None and isinstance(mod, HFConv1D):
                in_f, out_f = mod.nx, mod.nf
                W = mod.weight.data.T.contiguous()
                b = mod.bias.data if mod.bias is not None else None
            else:
                continue
            if not any(p in name for p in self.name_patterns):
                continue
            layer = GenerativeLayer(
                in_f, out_f, W, b, enabled=self.enabled,
                law_hidden=self.law_hidden, mode=self.mode,
                orig_layer=mod if self.mode == "residual" else None,
            )
            set_by_name(self.model, name, layer)
            self._layers[name] = layer
            replaced += 1

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def set_top_k(self, k: Optional[int]) -> None:
        for layer in self._layers.values():
            layer.set_top_k(k)

    def collect_codes(self) -> Dict[str, torch.Tensor]:
        return {n: l._last_code.clone()
                for n, l in self._layers.items()
                if l._last_code is not None}

    def describe(self) -> Dict[str, object]:
        return {
            "primitives": self.enabled,
            "replaced_layers": len(self._layers),
            "base_model": type(self.model).__name__,
            "mode": self.mode,
            "per_layer": {n: l.describe() for n, l in self._layers.items()},
        }
