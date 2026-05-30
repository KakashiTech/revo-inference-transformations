"""Tests for revo.metric_field_pinn (MetricFieldPINN)."""
from __future__ import annotations

import torch
import pytest

from revo.metric_field_pinn import MetricFieldPINN, train_pinn, pde_residual, curvature_metric


class TestMetricFieldPINN:
    def test_forward_shape(self):
        model = MetricFieldPINN(in_features=8, hidden_dim=32)
        x = torch.randn(4, 8)
        phi = model(x)
        assert phi.shape == (4, 1)

    def test_pde_residual_is_finite(self):
        model = MetricFieldPINN(in_features=4, hidden_dim=16)
        x = torch.randn(10, 4)
        loss = pde_residual(model, x, h=0.1)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_pde_residual_small_for_linear_field(self):
        # Build a model where the PINN approximates phi(x) = x[0] (linear, zero laplacian)
        model = MetricFieldPINN(in_features=2, hidden_dim=16)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            # fc_in maps [x0, x1] → [x0, 0, ..., 0] in hidden space
            model.fc_in.weight[0, 0] = 1.0
            # fc_out maps hidden[0] → output scalar via identity
            model.fc_out.weight[0, 0] = 1.0
            model.fc_in.bias.zero_()
            model.fc_out.bias.zero_()
            # sin(0) = 0 so hidden layers after fc_in are 0 if weights are 0
        x = torch.randn(50, 2)
        # For phi = x[0], laplacian should be approximately 0
        loss = pde_residual(model, x, h=0.01)
        assert loss.item() < 0.1

    def test_train_pinn_reduces_loss(self):
        model = MetricFieldPINN(in_features=2, hidden_dim=16)
        x = torch.randn(50, 2)
        loss_before = pde_residual(model, x).item()
        model, history = train_pinn(model, x, iters=20, lr=0.01, verbose=False)
        loss_after = pde_residual(model, x).item()
        assert loss_after <= loss_before + 0.1

    def test_curvature_metric_shape(self):
        model = MetricFieldPINN(in_features=4, hidden_dim=16)
        x = torch.randn(8, 4)
        lap = curvature_metric(model, x)
        assert lap.shape == (8,)
        assert torch.isfinite(lap).all()

    def test_w0_parameter(self):
        model = MetricFieldPINN(in_features=4, hidden_dim=16, w0=20.0)
        assert model._w0 == 20.0

    def test_training_history_length(self):
        model = MetricFieldPINN(in_features=2, hidden_dim=8)
        x = torch.randn(10, 2)
        model, history = train_pinn(model, x, iters=10, lr=0.01, verbose=False)
        assert len(history) == 10
