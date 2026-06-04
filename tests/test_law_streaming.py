"""Tests for revo/law_streaming.py — law generate and forward."""

from __future__ import annotations

import gc
import os

import pytest
import torch

from revo._utils import load_model_tokenizer
from revo.law_streaming import (
    build_law, pretrain_law, law_generate, law_stream_forward,
    _extract_svd_targets, _save_block_templates, law_info,
)
from revo.streaming import _free_all_block_weights

MODEL = os.environ.get("TEST_MODEL", "sshleifer/tiny-gpt2")


@pytest.fixture(scope="function")
def model_and_tok():
    m, t = load_model_tokenizer(MODEL)
    yield m, t
    del m
    gc.collect()


@pytest.fixture(scope="module")
def pretrained_law():
    m, _ = load_model_tokenizer(MODEL)
    law = build_law(m, rank=8, small_dim=4, hidden_dim=64)
    targets = _extract_svd_targets(m, rank=8)
    pretrain_law(law, targets, steps=30, lr=1e-3, verbose=False)
    del m
    gc.collect()
    return law


def test_law_build():
    model, _ = load_model_tokenizer(MODEL)
    law = build_law(model, rank=8, small_dim=4, hidden_dim=64)
    info = law_info(law)
    assert info["rank"] == 8
    assert info["small_dim"] == 4
    assert info["n_heads"] > 0
    assert info["n_layers"] > 0


def test_law_pretrain():
    model, _ = load_model_tokenizer(MODEL)
    law = build_law(model, rank=8, small_dim=4, hidden_dim=64)
    targets = _extract_svd_targets(model, rank=8)
    losses = pretrain_law(law, targets, steps=10, lr=1e-3, verbose=False)
    assert len(losses) == 10
    assert losses[-1] < losses[0] + 0.1


def test_law_generate_text(pretrained_law):
    model, tokenizer = load_model_tokenizer(MODEL)
    law = pretrained_law
    templates = _save_block_templates(model)
    _free_all_block_weights(model)

    text, meta = law_generate(
        model, tokenizer, "Hello", law,
        max_new_tokens=5,
        block_templates=templates, verbose=False,
    )
    assert len(text) > len("Hello")
    assert meta["n_new_tokens"] == 5
    assert meta["mean_time_per_token_s"] > 0
    assert meta["use_law"] is True


def test_law_generate_greedy(pretrained_law):
    model, tokenizer = load_model_tokenizer(MODEL)
    law = pretrained_law
    templates = _save_block_templates(model)
    _free_all_block_weights(model)

    text1, _ = law_generate(
        model, tokenizer, "The", law,
        max_new_tokens=5, temperature=0.0,
        block_templates=templates, verbose=False,
    )
    text2, _ = law_generate(
        model, tokenizer, "The", law,
        max_new_tokens=5, temperature=0.0,
        block_templates=templates, verbose=False,
    )
    assert text1 == text2, "Greedy (temp=0) must be deterministic"


def test_law_generate_without_free(pretrained_law):
    model, tokenizer = load_model_tokenizer(MODEL)
    law = pretrained_law
    text, meta = law_generate(
        model, tokenizer, "Hello", law,
        max_new_tokens=3, temperature=0.0,
        block_templates=None, verbose=False,
    )
    assert meta["n_new_tokens"] == 3
    assert meta["use_law"] is True


def test_law_generate_eos_no_crash(pretrained_law):
    model, tokenizer = load_model_tokenizer(MODEL)
    law = pretrained_law
    text, meta = law_generate(
        model, tokenizer, "<|endoftext|>", law,
        max_new_tokens=3,
        block_templates=None, verbose=False,
    )
    assert meta["n_new_tokens"] > 0
    # Generation with EOS in prompt should not crash


def test_law_stream_forward_preserved(pretrained_law):
    model, tokenizer = load_model_tokenizer(MODEL)
    law = pretrained_law
    text = "Hello world"
    enc = tokenizer(text, return_tensors="pt")
    logits = law_stream_forward(model, enc.input_ids, law, use_cache=False)
    assert logits is not None
    assert logits.shape[0] == 1
    assert logits.shape[2] == model.config.vocab_size


def test_cognitive_law_modulation():
    """Cognitive law must produce different weights for different hidden states."""
    model, _ = load_model_tokenizer(MODEL)
    law = build_law(model, rank=8, small_dim=4, hidden_dim=32, cognitive=True)

    x_a = torch.randn(1, 5, model.config.n_embd)
    x_b = torch.randn(1, 5, model.config.n_embd)

    out_a = law(0, hidden_state=x_a)
    out_b = law(0, hidden_state=x_b)

    key = list(out_a.keys())[0]
    diff = (out_a[key][0] - out_b[key][0]).abs().max().item()
    assert diff > 1e-6, "Cognitive modulation should produce different weights"
    assert law.cognitive is True


def test_cognitive_law_deterministic():
    """Same hidden state must produce same weights."""
    model, _ = load_model_tokenizer(MODEL)
    law = build_law(model, rank=8, small_dim=4, hidden_dim=32, cognitive=True)

    x = torch.randn(1, 5, model.config.n_embd)
    out1 = law(0, hidden_state=x)
    out2 = law(0, hidden_state=x)

    key = list(out1.keys())[0]
    diff = (out1[key][0] - out2[key][0]).abs().max().item()
    assert diff < 1e-6, "Same hidden state should produce identical weights"


def test_cognitive_law_backward_compat():
    """Cognitive law without hidden_state must behave like static law."""
    model, _ = load_model_tokenizer(MODEL)
    law = build_law(model, rank=8, small_dim=4, hidden_dim=32, cognitive=True)
    static = build_law(model, rank=8, small_dim=4, hidden_dim=32, cognitive=False)

    # Copy weights from static to cognitive (for same base)
    law.load_state_dict(static.state_dict(), strict=False)

    out_cog = law(0)  # no hidden_state
    out_static = static(0)

    key = list(out_static.keys())[0]
    diff = (out_cog[key][0] - out_static[key][0]).abs().max().item()
    assert diff < 1e-6, "Cognitive law without hidden_state should match static"
