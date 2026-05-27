"""Metric Field PINN — Physics-Informed Neural Network for information space.

Implements I.1 from GRH.md: a SIREN-like MLP that models the metric of
information space by minimising second-derivative PDE residuals on latent
coordinates (token embeddings).  This is a smoothness regulariser; the
"field equation" is approximated by the finite-difference Laplacian of the
learned scalar field.

CPU-only, pure PyTorch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class MetricFieldPINN(nn.Module):
    """SIREN-like MLP that outputs a scalar "metric potential" per input point.

    Architecture:  Linear(in,hidden) - sin(w0*) - Linear(hidden,hidden) - sin -
    Linear(hidden,hidden) - sin - Linear(hidden,hidden) - sin - Linear(hidden,1)
    """

    def __init__(self, in_features: int = 64, hidden_dim: int = 256, w0: float = 10.0) -> None:
        super().__init__()
        self.in_features = in_features
        self._w0 = w0

        self.fc_in = nn.Linear(in_features, hidden_dim)
        self.fc_h1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_h2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_h3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        bound_in = 1.0 / in_features
        nn.init.uniform_(self.fc_in.weight, -bound_in, bound_in)
        if self.fc_in.bias is not None:
            nn.init.uniform_(self.fc_in.bias, -bound_in, bound_in)

        for layer in (self.fc_h1, self.fc_h2, self.fc_h3, self.fc_out):
            bound_h = math.sqrt(6.0 / (layer.weight.shape[0] + layer.weight.shape[1]))
            nn.init.uniform_(layer.weight, -bound_h, bound_h)
            if layer.bias is not None:
                fan_in = layer.weight.shape[1]
                nn.init.uniform_(layer.bias, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sin(self._w0 * self.fc_in(x))
        x = torch.sin(self.fc_h1(x))
        x = torch.sin(self.fc_h2(x))
        x = torch.sin(self.fc_h3(x))
        return self.fc_out(x)


def _finite_diff_laplacian_and_grad(
    model: nn.Module, x: torch.Tensor, h: float
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    N, D = x.shape
    phi_0 = model(x).squeeze(-1)

    eye = torch.eye(D, device=x.device, dtype=x.dtype)
    x_pos = x.unsqueeze(1) + h * eye.unsqueeze(0)
    x_neg = x.unsqueeze(1) - h * eye.unsqueeze(0)

    phi_pos = model(x_pos.reshape(-1, D)).reshape(N, D)
    phi_neg = model(x_neg.reshape(-1, D)).reshape(N, D)

    d2 = (phi_pos - 2.0 * phi_0.unsqueeze(-1) + phi_neg) / (h * h)
    lap = d2.sum(dim=1)

    grad = (phi_pos - phi_neg) / (2.0 * h)
    grad_norm_sq = (grad ** 2).sum(dim=1)

    return lap, grad_norm_sq


def pde_residual(
    model: nn.Module,
    x: torch.Tensor,
    h: float = 0.1,
    grad_penalty: float = 0.0,
) -> torch.Tensor:
    lap, grad_norm_sq = _finite_diff_laplacian_and_grad(model, x, h=h)
    loss = (lap ** 2).mean()
    if grad_penalty > 0.0:
        loss = loss + grad_penalty * grad_norm_sq.mean()
    return loss


def train_pinn(
    model: MetricFieldPINN,
    embeddings: torch.Tensor,
    iters: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
    clip_norm: float = 1.0,
    scheduler: Optional[str] = None,
    scheduler_patience: int = 100,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    weight_decay: float = 0.0,
    grad_penalty: float = 0.0,
    fd_h: float = 0.1,
) -> Tuple[MetricFieldPINN, List[float]]:
    device = torch.device("cpu")
    model.to(device)
    embeddings = embeddings.to(device).detach()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []

    if scheduler == "plateau":
        scheduler_obj: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=min_lr,
            )
        )
    elif scheduler == "cosine":
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iters, eta_min=min_lr,
        )
    else:
        scheduler_obj = None

    model.train()
    for i in range(iters):
        optimizer.zero_grad()
        loss = pde_residual(model, embeddings, h=fd_h, grad_penalty=grad_penalty)
        loss.backward()
        if clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        loss_val = loss.item()
        history.append(loss_val)
        if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_obj.step(loss_val)
        elif scheduler_obj is not None:
            scheduler_obj.step()
        if verbose and (i + 1) % 50 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  iter {i+1:4d}/{iters}  loss={loss_val:.6e}  lr={cur_lr:.2e}")

    return model, history


def curvature_metric(model: nn.Module, embeddings: torch.Tensor, h: float = 0.01) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        lap, _ = _finite_diff_laplacian_and_grad(model, embeddings, h=h)
    return lap
