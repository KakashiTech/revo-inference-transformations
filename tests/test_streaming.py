"""Tests for revo/streaming.py — correctness and memory guarantees."""

from __future__ import annotations

import gc
import os
import tempfile

import pytest
import torch

from revo._utils import load_model_tokenizer, free_memory_trim
from revo.streaming import (
    shard_model, stream_forward, stream_generate,
    compare_memory, _n_layers, _detect_arch,
    _free_all_block_weights, _restore_all_blocks,
)

MODEL = os.environ.get("TEST_MODEL", "sshleifer/tiny-gpt2")


@pytest.fixture(scope="module")
def model_and_tok():
    m, t = load_model_tokenizer(MODEL)
    yield m, t
    del m
    gc.collect()


@pytest.fixture(scope="module")
def shard_dir(model_and_tok):
    model, _ = model_and_tok
    d = tempfile.mkdtemp(prefix="revo_test_shards_")
    shard_model(model, d)
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_detect_arch(model_and_tok):
    model, _ = model_and_tok
    arch = _detect_arch(model)
    assert arch in ("gpt2", "llama", "unknown")


def test_n_layers(model_and_tok):
    model, _ = model_and_tok
    n = _n_layers(model)
    assert n > 0
    assert isinstance(n, int)


def test_shard_sizes(model_and_tok, shard_dir):
    model, _ = model_and_tok
    n_layers = _n_layers(model)
    for i in range(n_layers):
        path = os.path.join(shard_dir, f"layer_{i}.pt")
        assert os.path.isfile(path), f"Missing shard for layer {i}"
        w = torch.load(path, map_location="cpu", weights_only=True)
        assert len(w) > 0, f"Empty shard for layer {i}"
    shared_path = os.path.join(shard_dir, "shared.pt")
    assert os.path.isfile(shared_path), "Missing shared shard"


def test_stream_forward_correctness(model_and_tok, shard_dir):
    """Streaming forward must produce bit-identical logits to full forward."""
    model, tokenizer = model_and_tok
    device = next(model.parameters()).device
    text = "Hello, this is a test of weight streaming."
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    with torch.no_grad():
        logits_full = model(input_ids).logits
    logits_stream = stream_forward(model, input_ids, shard_dir, use_cache=False, restore=True)

    diff = (logits_full - logits_stream).abs().max().item()
    assert diff == 0.0, f"Logit mismatch: max diff = {diff}"
    assert logits_full.shape == logits_stream.shape


def test_stream_generate(model_and_tok, shard_dir):
    """Streaming generation must produce valid text."""
    model, tokenizer = model_and_tok
    prompt = "The meaning of"
    text, meta = stream_generate(
        model, tokenizer, prompt, shard_dir,
        max_new_tokens=3, temperature=0.9, verbose=False,
    )
    assert len(text) > len(prompt)
    assert meta["n_new_tokens"] > 0
    assert meta["mean_time_per_token_s"] > 0


def test_freedom_cycles(model_and_tok, shard_dir):
    """Free → restore → free → restore must be idempotent."""
    model, _ = model_and_tok
    n_layers = _n_layers(model)

    _free_all_block_weights(model)
    _restore_all_blocks(model, shard_dir)

    param_count_1 = sum(p.numel() for p in model.parameters())

    _free_all_block_weights(model)
    _restore_all_blocks(model, shard_dir)

    param_count_2 = sum(p.numel() for p in model.parameters())
    assert param_count_1 == param_count_2, "Restored param count changed"


def test_memory_savings_theory(model_and_tok, shard_dir):
    """Theoretical memory savings must be positive."""
    model, tokenizer = model_and_tok
    texts = ["Test memory."]
    result = compare_memory(model, tokenizer, shard_dir, texts, max_length=16)
    assert result["theory"]["savings_pct"] > 0
    assert result["theory"]["full_weights_mb"] > result["theory"]["stream_peak_weights_mb"]


def test_multi_layer(model_and_tok, shard_dir):
    """Streaming forward must work on all layers."""
    model, tokenizer = model_and_tok
    n_layers = _n_layers(model)
    assert n_layers >= 1

    device = next(model.parameters()).device
    text = "Layer test"
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    logits = stream_forward(model, input_ids, shard_dir, use_cache=False, restore=True)
    assert logits is not None
    assert logits.shape[0] == 1
    assert logits.shape[2] == model.config.vocab_size
