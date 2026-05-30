"""Tests for revo.primitiva_router (token-conditional primitive selection)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.primitiva_router import (
    PrimitiveModel,
    PrimitiveSelector,
    PrimitiveRouter,
    DensePrimitive,
    CirculantPrimitive,
    WDMPrimitive,
    HolographyPrimitive,
    LowRankPrimitive,
    _nearest_circulant_first_column,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_model():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(32, 32)
            self.mlp = nn.Linear(32, 32)

    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleList([Block() for _ in range(2)])
            self.lm_head = nn.Linear(32, 64)
        def forward(self, x):
            for b in self.h:
                x = b.attn(x)
                x = b.mlp(x)
            return self.lm_head(x)
    return T()


# ─── Primitive tests ──────────────────────────────────────────────────────────

class TestPrimitives:
    def test_dense_shape(self):
        W = torch.randn(16, 32)
        b = torch.randn(16)
        p = DensePrimitive(W, b)
        x = torch.randn(4, 32)
        y = p(x)
        assert y.shape == (4, 16)

    def test_circulant_square_required(self):
        W = torch.randn(16, 16)
        p = CirculantPrimitive(W, None)
        x = torch.randn(2, 16)
        y = p(x)
        assert y.shape == (2, 16)
        assert torch.isfinite(y).all()

    def test_wdm_shape(self):
        W = torch.randn(16, 16)
        p = WDMPrimitive(W, None, bands=2)
        x = torch.randn(2, 16)
        y = p(x)
        assert y.shape == (2, 16)

    def test_holography_shape(self):
        W = torch.randn(16, 32)
        b = torch.randn(16)
        p = HolographyPrimitive(W, b, boundary_dim=4)
        x = torch.randn(2, 32)
        y = p(x)
        assert y.shape == (2, 16)
        assert torch.isfinite(y).all()

    def test_lowrank_shape(self):
        W = torch.randn(16, 32)
        b = torch.randn(16)
        p = LowRankPrimitive(W, b, rank=4)
        x = torch.randn(2, 32)
        y = p(x)
        assert y.shape == (2, 16)

    def test_dense_matches_linear(self):
        W = torch.randn(16, 32)
        b = torch.randn(16)
        p = DensePrimitive(W, b)
        lin = nn.Linear(32, 16)
        lin.weight.data = W.clone()
        lin.bias.data = b.clone()
        x = torch.randn(4, 32)
        y1 = p(x)
        y2 = lin(x)
        assert torch.allclose(y1, y2, atol=1e-6)


class TestPrimitiveRouter:
    def test_soft_selection(self):
        router = PrimitiveRouter(16, 4)
        x = torch.randn(3, 16)
        w = router(x, temperature=1.0, hard=False)
        assert w.shape == (3, 4)
        assert torch.allclose(w.sum(dim=-1), torch.ones(3))

    def test_hard_selection(self):
        router = PrimitiveRouter(16, 4)
        router.eval()
        x = torch.randn(3, 16)
        w = router(x, temperature=1.0, hard=True)
        assert w.shape == (3, 4)
        assert (w.sum(dim=-1) == 1).all()
        assert ((w == 0) | (w == 1)).all()

    def test_gumbel_training(self):
        router = PrimitiveRouter(16, 4)
        router.train()
        x = torch.randn(3, 16)
        w = router(x, temperature=1.0, hard=True)
        assert w.shape == (3, 4)


class TestPrimitiveSelector:
    def _make_sel(self, in_f, out_f, enabled, **kw):
        lin = nn.Linear(in_f, out_f)
        return PrimitiveSelector(in_f, out_f, lin.weight.data, lin.bias.data,
                                 enabled=enabled, **kw)

    def test_replace_linear(self):
        sel = self._make_sel(32, 32, enabled=["dense", "circulant"])
        x = torch.randn(2, 32)
        y = sel(x)
        assert y.shape == (2, 32)
        assert sel._last_weights is not None
        assert sel._last_weights.shape == (2, 2)

    def test_single_primitive_equals_linear(self):
        lin = nn.Linear(16, 16)
        sel = PrimitiveSelector(16, 16, lin.weight.data, lin.bias.data,
                                enabled=["dense"],
                                router_temperature=1.0, router_hard=False)
        x = torch.randn(4, 16)
        y_ref = lin(x)
        y = sel(x)
        assert torch.allclose(y, y_ref, atol=1e-5)

    def test_router_weights_normalize(self):
        sel = self._make_sel(32, 32, enabled=["dense", "circulant", "wdm"])
        x = torch.randn(4, 32)
        _ = sel(x)
        w = sel._last_weights
        assert w is not None
        assert torch.allclose(w.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_skip_nonsquare_circulant(self):
        sel = self._make_sel(32, 16, enabled=["dense", "circulant", "lowrank"])
        names = list(sel.primitives.keys())
        assert "circulant" not in names
        assert "dense" in names


class TestPrimitiveModel:
    def test_forward_shape(self, tiny_model):
        m = PrimitiveModel(tiny_model, enabled=["dense", "circulant"])
        x = torch.randn(2, 32)
        out = m(x)
        assert out.shape == (2, 64)

    def test_collect_router_weights(self, tiny_model):
        m = PrimitiveModel(tiny_model, enabled=["dense", "circulant", "wdm"])
        x = torch.randn(2, 32)
        _ = m(x)
        w = m.collect_router_weights()
        assert len(w) == 4  # 2 blocks × 2 layers (attn, mlp)
        for name, wt in w.items():
            assert wt.shape[1] == 3  # 3 primitives

    def test_differentiable(self, tiny_model):
        m = PrimitiveModel(tiny_model, enabled=["dense", "circulant"], router_hard=False)
        x = torch.randn(2, 32)
        out = m(x)
        loss = out.mean()
        loss.backward()
        router_grads = sum(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in m.named_parameters() if "router" in n
        )
        assert router_grads > 0

    def test_gradient_flow_through_primitives(self, tiny_model):
        m = PrimitiveModel(tiny_model, enabled=["dense", "circulant", "lowrank"])
        x = torch.randn(2, 32)
        out = m(x)
        out.mean().backward()
        prim_grads = sum(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in m.named_parameters() if "primitive" in n
        )
        assert prim_grads > 0

    def test_describe(self, tiny_model):
        m = PrimitiveModel(tiny_model, enabled=["dense", "circulant"])
        d = m.describe()
        assert d["replaced_layers"] == 4
        assert "dense" in d["primitives"]
        assert "circulant" in d["primitives"]

    def test_inference_hard_selection(self, tiny_model):
        m = PrimitiveModel(tiny_model, enabled=["dense", "circulant", "wdm", "holography", "lowrank"])
        m.eval()
        x = torch.randn(2, 32)
        out = m(x)
        assert out.shape == (2, 64)
        w = m.collect_router_weights()
        for wt in w.values():
            assert ((wt == 0) | (wt == 1)).all(), "hard selection should be one-hot"

    def test_all_primitives_finite(self, tiny_model):
        m = PrimitiveModel(tiny_model, enabled=["dense", "circulant", "wdm", "holography", "lowrank"])
        m.eval()
        x = torch.randn(2, 32)
        out = m(x)
        assert torch.isfinite(out).all()

    def test_replaces_all_linears(self, tiny_model):
        n_linear = sum(1 for _ in tiny_model.modules() if isinstance(_, nn.Linear))
        m = PrimitiveModel(tiny_model, enabled=["dense"])
        # lm_head is skipped, but attn+mlp per block are replaced
        replaced = m.describe()["replaced_layers"]
        assert replaced == n_linear - 1  # all except lm_head

    def test_no_crash_with_all_primitives(self, tiny_model):
        m = PrimitiveModel(
            tiny_model,
            enabled=["dense", "circulant", "wdm", "holography", "lowrank"],
            router_hard=False,
        )
        x = torch.randn(1, 32)
        out = m(x)
        assert out.shape == (1, 64)
        assert torch.isfinite(out).all()


class TestTopKSparse:
    def test_top_k_none_computes_all(self):
        lin = nn.Linear(16, 16)
        sel = PrimitiveSelector(16, 16, lin.weight.data, lin.bias.data,
                                enabled=["dense", "circulant", "wdm"],
                                top_k=None)
        x = torch.randn(4, 16)
        y = sel(x)
        assert y.shape == (4, 16)
        assert torch.isfinite(y).all()
        # All 3 primitives contributed (weights sum to 1)
        assert sel._last_weights is not None and sel._last_weights.shape[-1] == 3

    def test_top_k_1_selects_one(self):
        lin = nn.Linear(16, 16)
        sel = PrimitiveSelector(16, 16, lin.weight.data, lin.bias.data,
                                enabled=["dense", "circulant", "wdm"],
                                top_k=1)
        x = torch.randn(8, 16)
        y = sel(x)
        assert y.shape == (8, 16)

    def test_top_k_2_selects_two(self):
        lin = nn.Linear(16, 16)
        sel = PrimitiveSelector(16, 16, lin.weight.data, lin.bias.data,
                                enabled=["dense", "circulant", "wdm"],
                                top_k=2)
        x = torch.randn(8, 16)
        y = sel(x)
        assert y.shape == (8, 16)
        assert torch.isfinite(y).all()

    def test_top_k_1_matches_dense_when_one_primitive(self):
        lin = nn.Linear(16, 16)
        x = torch.randn(4, 16)
        # Only 1 primitive enabled → top_k=1 should match dense (top_k=None)
        y_ref = PrimitiveSelector(16, 16, lin.weight.data, lin.bias.data,
                                  enabled=["dense"], top_k=None)(x)
        y_sp = PrimitiveSelector(16, 16, lin.weight.data, lin.bias.data,
                                 enabled=["dense"], top_k=1)(x)
        assert torch.allclose(y_ref, y_sp, atol=1e-5)

    def test_set_top_k_runtime(self):
        lin = nn.Linear(16, 16)
        sel = PrimitiveSelector(16, 16, lin.weight.data, lin.bias.data,
                                enabled=["dense", "circulant", "wdm"],
                                top_k=None)
        x = torch.randn(3, 16)
        y_all = sel(x)
        sel.set_top_k(1)
        y_one = sel(x)
        assert y_all.shape == y_one.shape == (3, 16)
        assert torch.isfinite(y_one).all()

    def test_gradient_flow_sparse(self):
        sel = PrimitiveSelector(8, 8, torch.randn(8, 8), torch.zeros(8),
                                enabled=["dense", "circulant", "wdm"],
                                top_k=2)
        x = torch.randn(4, 8, requires_grad=True)
        y = sel(x)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        # Router params should also have gradients
        router_grad = sum(p.grad is not None and p.grad.abs().sum() > 0
                          for n, p in sel.named_parameters() if 'router' in n)
        assert router_grad > 0

    def test_top_k_clamps_to_available(self):
        sel = PrimitiveSelector(16, 16, torch.randn(16, 16), None,
                                enabled=["dense", "circulant"],
                                top_k=100)  # larger than available
        assert sel._top_k == 100  # stored as-is, clamped in forward
        x = torch.randn(2, 16)
        y = sel(x)
        assert y.shape == (2, 16)

    def test_sparse_model_level(self, tiny_model):
        m = PrimitiveModel(tiny_model, enabled=["dense", "circulant", "wdm"],
                           top_k=2)
        x = torch.randn(2, 32)
        out = m(x)
        assert out.shape == (2, 64)
        m.set_top_k(1)
        out2 = m(x)
        assert out2.shape == (2, 64)

    def test_sparse_batch_variation(self):
        """Different tokens in same batch can select different primitives."""
        sel = PrimitiveSelector(8, 8, torch.randn(8, 8), torch.zeros(8),
                                enabled=["dense", "circulant", "wdm"],
                                top_k=1)
        x = torch.randn(16, 8)
        _ = sel(x)
        w = sel._last_weights
        assert w is not None
        # At least 2 different primitives selected across 16 tokens
        unique = w.argmax(dim=-1).unique()
        assert len(unique) > 1, f"Only {len(unique)} primitive selected for all tokens"
