from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _near_factors(n: int) -> Tuple[int, int]:
    a = int(torch.sqrt(torch.tensor(float(max(1, n))))).item()
    for d in range(0, 1024):
        x = a + d
        if n % x == 0:
            return x, n // x
        y = max(1, a - d)
        if n % y == 0:
            return y, n // y
    return n, 1


@torch.no_grad()
def tt2_svd_from_linear(module: nn.Linear, chi: int) -> Tuple[torch.Tensor, torch.Tensor]:
    W = module.weight.detach().to(device="cpu", dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = int(max(1, min(chi, U.shape[1])))
    U_r = (U[:, :r] * S[:r])
    V_r = Vh[:r, :]
    return U_r.to(device=module.weight.device, dtype=module.weight.dtype), V_r.to(device=module.weight.device, dtype=module.weight.dtype)


class TT2Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, chi: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.chi = int(chi)
        self.U = nn.Linear(self.chi, self.out_features, bias=bias, device=device, dtype=dtype)
        self.V = nn.Linear(self.in_features, self.chi, bias=False, device=device, dtype=dtype)

    @torch.no_grad()
    def set_cores(self, U: torch.Tensor, V: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        assert U.shape == (self.out_features, self.chi)
        assert V.shape == (self.chi, self.in_features)
        self.U.weight.copy_(U)
        if bias is not None and self.U.bias is not None:
            self.U.bias.copy_(bias)
        self.V.weight.copy_(V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.V(x))


@torch.no_grad()
def replace_linear_with_tt2(model: nn.Module, ranks: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    report: Dict[str, Tuple[int, int]] = {}

    def set_by_name(root: nn.Module, path: str, new_mod: nn.Module) -> None:
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            if p.isdigit():
                parent = getattr(parent, "_modules")[p]
            else:
                parent = getattr(parent, p)
        last = parts[-1]
        if last.isdigit():
            parent._modules[last] = new_mod
        else:
            setattr(parent, last, new_mod)

    for name, m in model.named_modules():
        if name in ranks and isinstance(m, nn.Linear) and m.weight.dim() == 2:
            r = int(ranks[name])
            device = m.weight.device
            dtype = m.weight.dtype
            U_r, V_r = tt2_svd_from_linear(m, r)
            bias = m.bias.detach().to(device=device, dtype=dtype) if m.bias is not None else None
            new_mod = TT2Linear(m.in_features, m.out_features, r, bias=(m.bias is not None), device=device, dtype=dtype)
            new_mod.set_cores(U_r, V_r, bias=bias)
            set_by_name(model, name, new_mod)
            report[name] = (m.weight.shape[0], r)
    return report
