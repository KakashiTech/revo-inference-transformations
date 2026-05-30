"""Tests for revo.activation_cache."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from revo.activation_cache import ActivationCache


class TinyBlock(nn.Module):
    """Mini transformer block compatible with activation_cache tests."""
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        # Simplified attention + MLP with residual
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        return (x,)


class TinyModel(nn.Module):
    def __init__(self, n_layers=4, d_model=16, vocab=100):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab, d_model),
            "wpe": nn.Embedding(vocab, d_model),
            "h": nn.ModuleList([TinyBlock(d_model) for _ in range(n_layers)]),
            "ln_f": nn.LayerNorm(d_model),
        })
        self.lm_head = nn.Linear(d_model, vocab)


def _forward_through(model, input_ids):
    hs = model.transformer.wte(input_ids)
    hs = hs + model.transformer.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
    for block in model.transformer.h:
        hs = block(hs)[0]
    hs = model.transformer.ln_f(hs)
    return model.lm_head(hs)


class TestActivationCache:
    def test_cache_forward_shape(self):
        model = TinyModel()
        cache = ActivationCache()
        input_ids = torch.randint(0, 100, (1, 5))
        logits, layers = cache.cache_forward(model, input_ids)
        assert logits.shape == (1, 5, 100)
        assert len(layers) == 4
        assert layers[0].shape == (1, 5, 16)

    def test_forward_from_matches_full(self):
        model = TinyModel()
        cache = ActivationCache()
        input_ids = torch.randint(0, 100, (1, 5))

        logits_full = _forward_through(model, input_ids)
        logits_full_cached, _ = cache.cache_forward(model, input_ids)
        assert torch.allclose(logits_full, logits_full_cached, atol=1e-5)

        logits_from_0 = cache.forward_from(model, start_layer=0)
        assert torch.allclose(logits_full, logits_from_0, atol=1e-5)

    def test_forward_from_layer_2_differs(self):
        model = TinyModel()
        cache = ActivationCache()
        input_ids = torch.randint(0, 100, (1, 5))

        logits_full, _ = cache.cache_forward(model, input_ids)
        logits_partial = cache.forward_from(model, start_layer=2)

        assert logits_partial.shape == logits_full.shape

    def test_forward_from_after_modification(self):
        model = TinyModel()
        cache = ActivationCache()
        input_ids = torch.randint(0, 100, (1, 5))

        cache.cache_forward(model, input_ids)

        old_weight = model.transformer.h[2].mlp[1].weight.data.clone()
        model.transformer.h[2].mlp[1].weight.data += 0.01

        logits_full_mod = _forward_through(model, input_ids)
        logits_cached_mod = cache.forward_from(model, start_layer=2)

        assert torch.allclose(logits_full_mod, logits_cached_mod, atol=1e-5)

        model.transformer.h[2].mlp[1].weight.data = old_weight

    def test_get_hidden_at(self):
        model = TinyModel()
        cache = ActivationCache()
        input_ids = torch.randint(0, 100, (1, 5))
        _, layers = cache.cache_forward(model, input_ids)

        h2 = cache.get_hidden_at(2)
        assert h2 is not None
        assert torch.allclose(h2, layers[2])

        h99 = cache.get_hidden_at(99)
        assert h99 is None

    def test_clear(self):
        model = TinyModel()
        cache = ActivationCache()
        input_ids = torch.randint(0, 100, (1, 5))
        cache.cache_forward(model, input_ids)
        assert cache.n_layers == 4
        cache.clear()
        assert cache.n_layers == 0

    def test_get_modified_range_lm_head_only(self):
        first, last = ActivationCache.get_modified_range(["lm_head"])
        assert first == 0
        assert last == 0

    def test_get_modified_range_mixed(self):
        first, last = ActivationCache.get_modified_range([
            "transformer.h.3.mlp.c_fc",
            "transformer.h.8.attn.c_attn",
            "lm_head",
        ])
        assert first == 3
        assert last == 8

    def test_get_modified_range_single_layer(self):
        first, last = ActivationCache.get_modified_range([
            "transformer.h.5.attn.c_proj",
        ])
        assert first == 5
        assert last == 5

    def test_n_layers_property(self):
        model = TinyModel()
        cache = ActivationCache()
        assert cache.n_layers == 0
        cache.cache_forward(model, torch.randint(0, 100, (1, 3)))
        assert cache.n_layers == 4
