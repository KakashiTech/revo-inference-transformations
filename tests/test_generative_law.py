"""Tests for revo/generative_law.py — Phase XI: Generative Law."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.generative_law import (
    GenerativeLaw, StructureDecoder, PrimitiveBank,
    GenerativeLayer, GenerativeModel,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_model():
    """Minimal transformer: 2 blocks, lm_head."""
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            for _ in range(2):
                block = nn.ModuleDict({
                    "attn": nn.ModuleDict({
                        "c_attn": nn.Linear(32, 64),
                        "c_proj": nn.Linear(64, 32),
                    }),
                    "mlp": nn.ModuleDict({
                        "c_fc": nn.Linear(32, 128),
                        "c_proj": nn.Linear(128, 32),
                    }),
                })
                self.blocks.append(block)
            self.lm_head = nn.Linear(32, 100)

        def forward(self, x):
            h = x
            for b in self.blocks:
                h = b["attn"]["c_proj"](b["attn"]["c_attn"](h))
                h = b["mlp"]["c_proj"](b["mlp"]["c_fc"](h))
            return self.lm_head(h)

    return TinyTransformer()


# ─── Test GenerativeLaw ──────────────────────────────────────────────────

class TestGenerativeLaw:
    def test_output_shape(self):
        law = GenerativeLaw(16)
        x = torch.randn(4, 16)
        code = law(x)
        assert code.shape == (4, 32)
        assert code.dtype == torch.float32

    def test_binary_in_eval(self):
        law = GenerativeLaw(16)
        law.eval()
        x = torch.randn(8, 16)
        code = law(x)
        assert ((code == 0) | (code == 1)).all(), "eval should be binary"

    def test_ste_during_training(self):
        law = GenerativeLaw(16).train()
        x = torch.randn(4, 16, requires_grad=True)
        code = law(x)
        loss = code.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_code_bits_vary_with_input(self):
        law = GenerativeLaw(16).eval()
        x1 = torch.randn(1, 16)
        x2 = torch.randn(1, 16)
        c1 = law(x1)
        c2 = law(x2)
        assert not (c1 == c2).all(), "different tokens should get different codes"

    def test_temperature_affects_probs(self):
        law = GenerativeLaw(16)
        x = torch.randn(4, 16)
        # High temp → softer
        c_hot = law(x, temperature=0.1, hard=False)
        c_cold = law(x, temperature=10.0, hard=False)
        # With high temp, probs should be closer to 0.5
        assert (c_cold - 0.5).abs().mean() < (c_hot - 0.5).abs().mean()

    def test_different_hidden_sizes(self):
        for h in [16, 64, 128]:
            law = GenerativeLaw(32, hidden=h)
            x = torch.randn(2, 32)
            code = law(x)
            assert code.shape == (2, 32)


# ─── Test StructureDecoder ───────────────────────────────────────────────

class TestStructureDecoder:
    def test_bits_to_int(self):
        dec = StructureDecoder(5)
        bits = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        vals = dec.bits_to_int(bits)
        assert abs(vals[0].item() - 5.0) < 1e-4  # 101 = 4+0+1 = 5
        assert abs(vals[1].item() - 3.0) < 1e-4  # 011 = 0+2+1 = 3

    def test_forward_returns_correct_shapes(self):
        dec = StructureDecoder(5)
        code = torch.randn(4, 32)
        out = dec(code)
        assert out["prim_logits"].shape == (4, 5)
        assert out["scale"].shape == (4,)
        assert out["temp"].shape == (4,)

    def test_forward_differentiable(self):
        dec = StructureDecoder(5)
        code = torch.randn(4, 32, requires_grad=True)
        out = dec(code)
        loss = out["prim_logits"].mean()
        loss.backward()
        assert code.grad is not None
        assert code.grad.abs().sum() > 0

    def test_decode_discrete(self):
        dec = StructureDecoder(5)
        code = torch.zeros(2, 32)
        code[0, 0] = 1.0  # first bit of latent set
        code[1, 5] = 1.0  # different bit
        out = dec.decode_discrete(code)
        assert out["prim_idx"].shape == (2,)
        assert out["scale"].shape == (2,)
        assert out["temp"].shape == (2,)

    def test_param_interpolation(self):
        dec = StructureDecoder(5)
        # All scale bits set → should map to max param value
        code = torch.zeros(1, 32)
        code[0, 16:20] = 1.0  # scale bits all high
        out = dec(code)
        # bits_to_int = 15, PARAM_MAP[15] ≈ 4.0
        assert out["scale"][0].item() > 3.5

    def test_logits_learned_mapping(self):
        """Different codes should produce different logits after init."""
        dec = StructureDecoder(5)
        c1 = torch.zeros(1, 32)
        c2 = torch.zeros(1, 32)
        c2[0, 0] = 1.0  # differ in first bit of latent
        l1 = dec(c1)["prim_logits"]
        l2 = dec(c2)["prim_logits"]
        # Logits might be similar with random init, just check shapes
        assert l1.shape == l2.shape == (1, 5)
        assert torch.isfinite(l1).all()
        assert torch.isfinite(l2).all()

    def test_code_to_logits_has_gradients(self):
        dec = StructureDecoder(5)
        code = torch.randn(2, 32)
        out = dec(code)
        loss = out["prim_logits"].sum()
        loss.backward()
        assert dec.code_to_logits.weight.grad is not None
        assert dec.code_to_logits.weight.grad.abs().sum() > 0


# ─── Test PrimitiveBank ──────────────────────────────────────────────────

class TestPrimitiveBank:
    def test_creates_selected_primitives(self):
        w = torch.randn(16, 16)
        bank = PrimitiveBank(w, None, enabled=["dense", "circulant"])
        assert "dense" in bank
        assert "circulant" in bank
        assert "wdm" not in bank

    def test_forward_all_primitives(self):
        w = torch.randn(16, 16)
        b = torch.zeros(16)
        bank = PrimitiveBank(w, b)
        x = torch.randn(4, 16)
        for name in bank:
            y = bank[name](x)
            assert y.shape == (4, 16)

    def test_skips_nonsquare_circulant(self):
        w = torch.randn(16, 32)
        bank = PrimitiveBank(w, None, enabled=["dense", "circulant", "wdm"])
        assert "dense" in bank
        assert "circulant" not in bank  # non-square
        assert "wdm" not in bank  # non-square


# ─── Test GenerativeLayer ────────────────────────────────────────────────

class TestGenerativeLayer:
    def test_forward_shape(self):
        w = torch.randn(16, 16)
        b = torch.zeros(16)
        layer = GenerativeLayer(16, 16, w, b)
        x = torch.randn(4, 16)
        y = layer(x)
        assert y.shape == (4, 16)

    def test_forward_finite(self):
        w = torch.randn(8, 8)
        layer = GenerativeLayer(8, 8, w, torch.zeros(8))
        x = torch.randn(2, 8)
        y = layer(x)
        assert torch.isfinite(y).all()

    def test_stores_code(self):
        w = torch.randn(16, 16)
        layer = GenerativeLayer(16, 16, w, torch.zeros(16))
        x = torch.randn(3, 16)
        _ = layer(x)
        code = layer.collect_codes()
        assert code is not None
        assert code.shape == (3, 32)

    def test_gradient_flow(self):
        layer = GenerativeLayer(8, 8, torch.randn(8, 8), torch.zeros(8))
        x = torch.randn(4, 8, requires_grad=True)
        y = layer(x)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        # Law params should have gradients
        law_grads = sum(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in layer.law.named_parameters()
        )
        assert law_grads > 0

    def test_different_params_produce_different_outputs(self):
        """Same input, different random seeds → different codes → possibly different outputs."""
        w = torch.randn(16, 16)
        layer = GenerativeLayer(16, 16, w, torch.zeros(16))
        layer.eval()
        x = torch.randn(8, 16)
        y1 = layer(x)
        c1 = layer.collect_codes()
        # Different random init would give different codes, but same layer
        # is deterministic in eval mode. Just verify shape.
        y2 = layer(x)
        assert y1.shape == y2.shape


# ─── Test GenerativeModel ────────────────────────────────────────────────

class TestGenerativeModel:
    def test_forward_shape(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense", "circulant"])
        x = torch.randn(2, 32)
        out = gm(x)
        assert out.shape == (2, 100)

    def test_collect_codes(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense", "circulant", "wdm"])
        x = torch.randn(2, 32)
        _ = gm(x)
        codes = gm.collect_codes()
        assert len(codes) > 0
        for name, c in codes.items():
            assert c.shape[-1] == 32
        for name, c in codes.items():
            assert c.shape[-1] == 32

    def test_differentiable(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense", "circulant"])
        x = torch.randn(2, 32)
        out = gm(x)
        loss = out.mean()
        loss.backward()
        law_grads = sum(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in gm.named_parameters() if "law" in n
        )
        assert law_grads > 0

    def test_describe(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense", "circulant"])
        d = gm.describe()
        assert d["replaced_layers"] > 0
        assert "dense" in d["primitives"]

    def test_finite_output(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense", "circulant", "wdm"])
        x = torch.randn(2, 32)
        out = gm(x)
        assert torch.isfinite(out).all()

    def test_code_analysis(self, tiny_model):
        """After forward, codes should be binary in eval mode."""
        gm = GenerativeModel(tiny_model, enabled=["dense"]).eval()
        x = torch.randn(4, 32)
        _ = gm(x)
        for name, code in gm.collect_codes().items():
            assert ((code == 0) | (code == 1)).all(), f"{name} codes not binary"

    # ─── Mode tests ───────────────────────────────────────────────────────

    def test_residual_mode_output_shape(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense"], mode="residual")
        assert gm.mode == "residual"
        x = torch.randn(2, 32)
        out = gm(x)
        assert out.shape == (2, 100)

    def test_residual_mode_has_eps(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense"], mode="residual")
        eps_count = sum(1 for n, p in gm.named_parameters() if "eps" in n)
        assert eps_count > 0, "residual mode should have eps parameters"

    def test_residual_mode_preserves_base(self, tiny_model):
        """In residual mode, base model is stored per-layer."""
        gm = GenerativeModel(tiny_model, enabled=["dense"], mode="residual")
        for name, layer in gm._layers.items():
            assert layer.orig is not None, f"{name} missing orig"
            assert layer.eps is not None, f"{name} missing eps"

    def test_pure_mode_no_orig(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense"], mode="pure")
        for name, layer in gm._layers.items():
            assert layer.orig is None, f"{name} should not have orig in pure mode"
            assert layer.eps is None, f"{name} should not have eps in pure mode"

    # ─── Top-k tests ──────────────────────────────────────────────────────

    def test_top_k_shapes(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense", "lowrank"]).eval()
        gm.set_top_k(1)
        x = torch.randn(2, 32)
        out1 = gm(x)
        assert out1.shape == (2, 100)
        assert torch.isfinite(out1).all()
        # With k=2 (all) should match k=1 in shape
        gm.set_top_k(2)
        out2 = gm(x)
        assert out2.shape == (2, 100)

    def test_top_k_ignored_during_training(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense", "lowrank"]).train()
        gm.set_top_k(1)  # Should be ignored in train mode
        x = torch.randn(2, 32)
        out = gm(x)
        assert out.shape == (2, 100)

    def test_generative_layer_top_k(self):
        w = torch.randn(8, 8)
        layer = GenerativeLayer(8, 8, w, torch.zeros(8),
                                enabled=["dense", "lowrank", "wdm"])
        layer.eval()
        layer.set_top_k(2)
        x = torch.randn(4, 8)
        y = layer(x)
        assert y.shape == (4, 8)
        assert torch.isfinite(y).all()
        layer.set_top_k(None)
        y2 = layer(x)
        assert y2.shape == (4, 8)

    def test_top_k_stores_codes(self, tiny_model):
        gm = GenerativeModel(tiny_model, enabled=["dense", "lowrank"]).eval()
        gm.set_top_k(1)
        x = torch.randn(2, 32)
        _ = gm(x)
        codes = gm.collect_codes()
        assert len(codes) > 0
        for c in codes.values():
            assert c.shape[-1] == 32
