from __future__ import annotations
from revo._logging import get_logger

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from revo.fft_kernel import nearest_circulant_first_column
from revo._utils import set_by_name, skip_tied_weights

def _infer_in_out(module: nn.Module) -> Tuple[int, int]:
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    if isinstance(b, torch.Tensor):
        if W.shape[0] == b.numel():
            return int(W.shape[1]), int(W.shape[0])
        if W.shape[1] == b.numel():
            return int(W.shape[0]), int(W.shape[1])
    return int(W.shape[1]), int(W.shape[0])


class WDMLinearWrap(nn.Module):
    """Block-diagonal circulant approximation across bands (WDM-like parallel subchannels).

    Requires in_features == out_features == N and N % bands == 0.
    Splits x into B bands and applies independent circulant kernels per band via FFT.
    """

    def __init__(self, base: nn.Module, bands: int = 2):
        super().__init__()
        assert hasattr(base, "weight") and getattr(base, "weight").dim() == 2
        self.base = base
        in_f, out_f = _infer_in_out(base)
        assert in_f == out_f, "WDM only supports square mappings"
        self.features = int(in_f)
        self.bands = int(max(1, bands))
        assert self.features % self.bands == 0, "features must be divisible by bands"
        self.band_size = self.features // self.bands
        device = base.weight.device
        dtype = base.weight.dtype
        # Learn per-band bias if base had bias
        b = getattr(base, "bias", None)
        if isinstance(b, torch.Tensor):
            self.bias = nn.Parameter(b.detach().clone().to(device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        # Compute per-band circulant first columns from block-diagonal of W (nearest)
        W = base.weight.detach()
        if isinstance(b, torch.Tensor) and W.shape[1] == b.numel():
            W = W.t()  # ensure [out,in]
        self.register_buffer("c_cols", torch.zeros(self.bands, self.band_size, device=device, dtype=dtype), persistent=False)
        with torch.no_grad():
            for bi in range(self.bands):
                start = bi * self.band_size
                stop = start + self.band_size
                W_block = W[start:stop, start:stop]
                c = nearest_circulant_first_column(W_block)
                self.c_cols[bi].copy_(c)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct block-diagonal circulant matrix (for downstream compatibility)."""
        B, S = self.c_cols.shape
        N = B * S
        dev = self.c_cols.device
        idx = torch.arange(S, device=dev)
        rows = (idx[:, None] - idx[None, :]) % S
        blocks = []
        for bi in range(B):
            blocks.append(self.c_cols[bi][rows.long()])
        return torch.block_diag(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., N]
        N = self.features
        B = self.bands
        S = self.band_size
        assert x.size(-1) == N
        xs = x.reshape(*x.shape[:-1], B, S)
        # FFT per band, multiply by per-band c spectrum, inverse FFT
        cfreq = torch.fft.rfft(self.c_cols, dim=-1)  # [B, S//2+1]
        xfreq = torch.fft.rfft(xs, dim=-1)  # [..., B, S//2+1]
        yfreq = xfreq * cfreq  # broadcasting
        ys = torch.fft.irfft(yfreq, n=S, dim=-1)
        y = ys.reshape(*x.shape[:-1], N)
        if self.bias is not None:
            y = y + self.bias
        return y


@torch.no_grad()
def replace_with_wdm(
    model: nn.Module,
    bands: int = 2,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, Tuple[int, int]]:
    pats = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    report: Dict[str, Tuple[int, int]] = {}

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
        if not isinstance(m, nn.Linear):
            continue
        try:
            if (input_embed_weight is not None) and (m.weight is input_embed_weight):
                continue
        except Exception:
            get_logger().warning("except Exception:")
        in_f, out_f = _infer_in_out(m)
        if in_f != out_f:
            continue
        if in_f % bands != 0:
            continue
        wrapper = WDMLinearWrap(m, bands=bands)
        set_by_name(model, name, wrapper)
        report[name] = (int(in_f), int(out_f))
    return report
