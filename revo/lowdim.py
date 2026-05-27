from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from revo._utils import orient_weight, set_by_name, skip_tied_weights

def _orient_weight_bias(module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor | None, int, int, bool]:
    W = getattr(module, "weight")
    b = getattr(module, "bias", None)
    assert isinstance(W, torch.Tensor) and W.dim() == 2
    b_det = b.detach() if isinstance(b, torch.Tensor) else None
    W_det = W.detach()
    if b_det is not None:
        if W_det.shape[0] == b_det.numel():
            return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False
        if W_det.shape[1] == b_det.numel():
            return W_det.t(), b_det, int(W_det.shape[0]), int(W_det.shape[1]), True
    return W_det, b_det, int(W_det.shape[1]), int(W_det.shape[0]), False


class LowDimAdapter(nn.Module):
    """Projects inputs to a low-dim subspace then applies a compressed linear.

    y ≈ (W @ V_d) (x @ V_d) + b, where columns of V_d span top principal components of inputs.
    V_d is stored as a buffer, not trained.
    """

    def __init__(self, base: nn.Module, V_d: torch.Tensor, dtype: torch.dtype | None = None):
        super().__init__()
        W_o, b_o, in_f, out_f, _ = _orient_weight_bias(base)
        assert V_d.dim() == 2 and V_d.shape[0] == in_f
        d = int(V_d.shape[1])
        self.in_features = int(in_f)
        self.out_features = int(out_f)
        self.d = d
        dev = W_o.device
        dt = dtype if dtype is not None else W_o.dtype
        self.register_buffer("Vd", V_d.to(device=dev, dtype=dt), persistent=False)  # [in, d]
        self.B = nn.Linear(d, out_f, bias=b_o is not None, device=dev, dtype=dt)
        # Initialize B to match W @ Vd
        with torch.no_grad():
            Wd = W_o.to(dt) @ self.Vd  # [out, d]
            self.B.weight.copy_(Wd)
            if b_o is not None:
                self.B.bias.copy_(b_o.to(device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in_dtype = x.dtype
        x_d = (x.to(self.Vd.dtype)) @ self.Vd  # [..., d]
        y = self.B(x_d.to(self.B.weight.dtype))    # [..., out]
        return y.to(x_in_dtype)


@torch.no_grad()
def consolidate_lowdim(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    name_patterns: Optional[List[str]] = None,
    frac: float = 0.98,
    max_length: int = 128,
    max_samples: int = 4096,
    dtype: str = "float32",
) -> Dict[str, Tuple[int, int, int]]:
    """Consolidate modules by projecting inputs to a PCA subspace of dimension d=floor(frac*in).

    Collect real inputs via forward_pre hooks. For each selected module with [out,in] weight:
    - Compute PCA basis V_d from inputs (top-d right singular vectors of X).
    - Replace module with LowDimAdapter using V_d.
    Returns report: name -> (in, out, d).
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
    hooks: List[torch.utils.hooks.RemovableHandle] = []  # type: ignore[name-defined]

    def _pre_hook(name: str):
        def fn(mod, inputs):
            try:
                x = inputs[0]
                if isinstance(x, torch.Tensor):
                    x = x.detach().to(device="cpu", dtype=torch.float32)
                    if x.dim() > 2:
                        x = x.view(-1, x.shape[-1])
                    buffers[name].append(x)
            except Exception:
                pass
        return fn

    for n, m in wanted.items():
        try:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp, _n=n: _pre_hook(_n)(mod, inp)))
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
            pass

    report: Dict[str, Tuple[int, int, int]] = {}

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

    for name, xs in buffers.items():
        if not xs:
            continue
        X = torch.cat(xs, dim=0)
        if X.shape[0] > max_samples:
            X = X[: max_samples]
        # PCA via SVD on centered inputs
        X = X - X.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        in_dim = int(Vh.shape[1])
        d = int(max(1, min(in_dim, int(in_dim * float(frac)))))
        V_d = Vh[:d, :].t().contiguous()  # [in, d]
        base = wanted[name]
        # Verify orientation dims
        W_o, b_o, in_f, out_f, _ = _orient_weight_bias(base)
        if int(in_f) != in_dim:
            # If shapes mismatch (rare), skip
            continue
        # choose dtype
        _dt = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype, W_o.dtype)
        adapter = LowDimAdapter(base, V_d, dtype=_dt)
        _set_by_name(model, name, adapter)
        report[name] = (int(in_f), int(out_f), int(d))

    return report
