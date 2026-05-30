"""Tests for revo.frequency (frequency band analysis)."""
from __future__ import annotations

import torch
import pytest

from revo.frequency import FreqConfig, _entropy_from_logits, _sequence_best_band_and_coherence


class TestFrequency:
    def test_default_bands(self):
        cfg = FreqConfig()
        assert "delta" in cfg.bands
        assert "gamma" in cfg.bands

    def test_entropy_from_logits_finite(self):
        logits = torch.randn(10)
        h = _entropy_from_logits(logits)
        assert h >= 0.0
        assert h < 10.0

    def test_entropy_peaked_small(self):
        logits = torch.tensor([10.0, 0.0, 0.0])
        h = _entropy_from_logits(logits)
        assert h < 1.0

    def test_entropy_flat_larger(self):
        logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
        h = _entropy_from_logits(logits)
        assert h > 1.0

    def test_sequence_best_band(self):
        logits_seq = torch.randn(20, 10)
        cfg = FreqConfig()
        band, coh = _sequence_best_band_and_coherence(logits_seq, cfg)
        assert isinstance(band, str)
        assert band in cfg.bands
        assert 0.0 <= coh <= 1.0

    def test_short_sequence_defaults_to_delta(self):
        logits_seq = torch.randn(2, 10)
        cfg = FreqConfig()
        band, coh = _sequence_best_band_and_coherence(logits_seq, cfg)
        assert band == "delta"
        assert coh == 0.0

    def test_custom_phases(self):
        cfg = FreqConfig(phases=16)
        assert cfg.phases == 16

    def test_custom_bands(self):
        cfg = FreqConfig(bands={"alpha": 4, "beta": 8})
        assert list(cfg.bands.keys()) == ["alpha", "beta"]
