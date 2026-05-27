from __future__ import annotations
from revo._logging import get_logger

from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn

from revo._utils import orient_weight, set_by_name, skip_tied_weights

def _orient_weight_bias(module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor | None, int, int, bool]:
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    if not isinstance(W, torch.Tensor) or W.dim() != 2:
        raise AssertionError("module.weight must be 2D tensor")
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    W_det = W.detach()
    if b_det is not None:
        if W_det.shape[0] == b_det.numel():
            return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False
        if W_det.shape[1] == b_det.numel():
            return W_det.t(), b_det, int(W_det.shape[0]), int(W_det.shape[1]), True
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False


@torch.no_grad()
def mi_fuse_outputs(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    name_patterns: Optional[List[str]] = None,
    threshold: float = 0.98,
    max_length: int = 128,
    max_samples: int = 4096,
) -> Dict[str, Dict[str, int]]:
    """Approximate MI-based fusion by tying highly correlated output units.

    - Collect module outputs across provided texts via forward hooks.
    - Compute Pearson correlation for output dimensions; translate to Gaussian-MI proxy.
    - For |rho| >= threshold, tie rows (outputs) by setting both to their average weights/bias.
    - Returns report name -> {out, fused_pairs}.
    """
    patterns = name_patterns or ["mlp", "c_fc", "c_proj", "attn"]
    wanted: Dict[str, nn.Module] = {
        n: m for n, m in model.named_modules()
        if any(p in n for p in patterns)
        and hasattr(m, "weight")
        and isinstance(getattr(m, "weight"), torch.Tensor)
        and getattr(m, "weight").dim() == 2
    }

    buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in wanted}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(name: str):
        def fn(mod, inp, out):
            try:
                y = out
                if isinstance(y, torch.Tensor):
                    y = y.detach().to(device="cpu", dtype=torch.float32)
                    if y.dim() > 2:
                        y = y.view(-1, y.shape[-1])
                    buffers[name].append(y)
            except Exception:
                get_logger().warning("except Exception:")
        return fn

    for n, m in wanted.items():
        try:
            hooks.append(m.register_forward_hook(_hook(n)))
        except Exception:
            continue

    model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            get_logger().warning("except Exception:")
    report: Dict[str, Dict[str, int]] = {}
    for name, parts in buffers.items():
        if not parts:
            continue
        Y = torch.cat(parts, dim=0)
        if Y.shape[0] > max_samples:
            Y = Y[: max_samples]
        # Y: [N, out]
        out_dim = int(Y.shape[1])
        if out_dim < 2:
            report[name] = {"out": out_dim, "fused_pairs": 0}
            continue
        # Standardize
        Y = Y - Y.mean(dim=0, keepdim=True)
        std = Y.std(dim=0, keepdim=True).clamp_min(1e-6)
        Y = Y / std
        # Correlation matrix
        C = (Y.t() @ Y) / max(1, Y.shape[0] - 1)
        # Greedy fuse pairs above threshold
        used = torch.zeros(out_dim, dtype=torch.bool)
        fused_pairs = 0
        for i in range(out_dim - 1):
            if used[i]:
                continue
            row = C[i]
            row[i] = 0.0
            # search partner j>i
            if i + 1 >= out_dim:
                continue
            seg = row[i + 1 :]
            if seg.numel() == 0:
                continue
            j_val, j_idx = torch.max(torch.abs(seg), dim=0)
            if j_val.item() >= float(threshold):
                j = int(i + 1 + j_idx.item())
                if j < out_dim and (not used[j]):
                    # Tie rows i and j by averaging weights and bias
                    mod = wanted[name]
                    W_o, b_o, in_f, out_f, transposed = _orient_weight_bias(mod)
                    wi = W_o[i].clone()
                    wj = W_o[j].clone()
                    w_avg = 0.5 * (wi + wj)
                    W_o[i].copy_(w_avg)
                    W_o[j].copy_(w_avg)
                    if isinstance(b_o, torch.Tensor):
                        bi = b_o[i].clone()
                        bj = b_o[j].clone()
                        b_avg = 0.5 * (bi + bj)
                        b_o[i].copy_(b_avg)
                        b_o[j].copy_(b_avg)
                    if transposed:
                        mod.weight.data.copy_(W_o.t().to(dtype=mod.weight.dtype, device=mod.weight.device))
                    else:
                        mod.weight.data.copy_(W_o.to(dtype=mod.weight.dtype, device=mod.weight.device))
                    if isinstance(b_o, torch.Tensor) and hasattr(mod, "bias") and isinstance(mod.bias, torch.Tensor):
                        mod.bias.data.copy_(b_o.to(dtype=mod.bias.dtype, device=mod.bias.device))
                    used[i] = True
                    used[j] = True
                    fused_pairs += 1
        report[name] = {"out": out_dim, "fused_pairs": int(fused_pairs)}

    return report
