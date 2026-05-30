"""Tests for revo.oscillatory_gating (OscillatoryHooks)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from revo.oscillatory_gating import OscillatoryHooks


class TestOscillatoryHooks:
    def test_attach_returns_modules(self):
        model = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )
        mgr = OscillatoryHooks(alpha=0.1, freq=0.5)
        selected = mgr.attach(model, name_patterns=["0", "1"], skip_lm_head=False)
        assert len(selected) >= 1

    def test_detach_clears(self):
        model = nn.Linear(4, 4)
        mgr = OscillatoryHooks()
        mgr.attach(model, name_patterns=[""], skip_lm_head=False)
        mgr.detach()
        assert len(mgr._hooks) == 0

    def test_phases_dict(self):
        model = nn.Linear(4, 4)
        mgr = OscillatoryHooks()
        mgr.attach(model, name_patterns=[""], skip_lm_head=False)
        phases = mgr.phases()
        assert isinstance(phases, dict)
        for v in phases.values():
            assert -4.0 <= v <= 4.0

    def test_step_tokens(self):
        mgr = OscillatoryHooks()
        assert mgr._t == 0.0
        mgr.step_tokens(5)
        assert mgr._t == 5.0

    def test_reset_time(self):
        mgr = OscillatoryHooks()
        mgr.step_tokens(10)
        mgr.reset_time()
        assert mgr._t == 0.0

    def test_phase_coherence(self):
        mgr = OscillatoryHooks()
        # With all phases same, coherence = 1
        mgr._phases = {"a": torch.tensor(0.0), "b": torch.tensor(0.0)}
        assert abs(mgr.phase_coherence() - 1.0) < 1e-6

    def test_kuramoto_step_changes_phases(self):
        mgr = OscillatoryHooks()
        mgr._phases = {"a": torch.tensor(0.0), "b": torch.tensor(1.0)}
        phis_before = [float(v.item()) for v in mgr._phases.values()]
        mgr.kuramoto(steps=100, kappa=2.0, dt=0.1)
        phis_after = [float(v.item()) for v in mgr._phases.values()]
        # Phases should have changed
        assert phis_after != phis_before

    def test_set_phase_offset(self):
        mgr = OscillatoryHooks()
        mgr._phases = {"test": torch.tensor(1.0)}
        mgr.set_phase_offset(0.5)
        assert abs(mgr._phases["test"].item() - 1.5) < 1e-6

    def test_trainable_parameters(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        mgr = OscillatoryHooks()
        mgr.attach(model, name_patterns=["0"], skip_lm_head=False)
        params = mgr.make_phases_trainable()
        assert len(params) >= 1
        assert all(isinstance(p, torch.nn.Parameter) for p in params)
