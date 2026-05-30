"""Tests for revo.hyperbolic (Poincaré ball operations)."""
from __future__ import annotations

import torch
import pytest

from revo.hyperbolic import project_to_ball, expmap0, logmap0, mobius_add, mobius_matvec


class TestHyperbolic:
    def test_project_to_ball_stays_inside(self):
        x = torch.randn(100, 8)
        c = 0.1
        projected = project_to_ball(x, c)
        norms = projected.norm(dim=-1)
        max_norm = (1.0 / (c ** 0.5)) - 1e-6
        assert (norms <= max_norm + 1e-4).all()

    def test_project_ball_identity_for_small(self):
        x = torch.randn(4) * 0.1
        c = 1.0
        projected = project_to_ball(x, c)
        assert torch.allclose(x, projected, atol=1e-6)

    def test_expmap0_logmap0_roundtrip(self):
        v = torch.randn(2, 8) * 0.3
        c = 0.1
        x = expmap0(v, c)
        v2 = logmap0(x, c)
        assert torch.allclose(v, v2, atol=1e-4)

    def test_expmap0_zero(self):
        v = torch.zeros(4)
        c = 0.1
        x = expmap0(v, c)
        assert torch.allclose(x, torch.zeros(4), atol=1e-6)

    def test_logmap0_zero(self):
        x = torch.zeros(4)
        c = 0.1
        v = logmap0(x, c)
        assert torch.allclose(v, torch.zeros(4), atol=1e-6)

    def test_mobius_add_commutative_small(self):
        # Möbius addition in Poincaré ball is approximately commutative for small x,y
        x = torch.randn(4) * 0.01
        y = torch.randn(4) * 0.01
        c = 0.1
        a = mobius_add(x, y, c)
        b = mobius_add(y, x, c)
        assert torch.allclose(a, b, atol=1e-4)

    def test_mobius_add_zero(self):
        x = torch.randn(4) * 0.2
        c = 0.1
        assert torch.allclose(mobius_add(x, torch.zeros(4), c), x, atol=1e-6)

    def test_mobius_matvec_output_shape(self):
        M = torch.randn(3, 4)
        x = torch.randn(2, 4) * 0.2
        c = 0.1
        y = mobius_matvec(M, x, c)
        assert y.shape == (2, 3)

    def test_mobius_matvec_c0_equals_linear(self):
        M = torch.randn(4, 4)
        x = torch.randn(2, 4)
        y_hyp = mobius_matvec(M, x, 0.0)
        y_lin = x @ M.T
        assert torch.allclose(y_hyp, y_lin, atol=1e-6)

    def test_expmap0_norm_relationship(self):
        v = torch.randn(4) * 0.5
        c = 0.2
        x = expmap0(v, c)
        x_norm = x.norm().item()
        assert x_norm < (1.0 / (c ** 0.5)) + 1e-4
