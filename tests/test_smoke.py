"""Smoke tests: verify all revo modules import and expose expected symbols."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import revo


def _list_modules():
    """Return all revo submodules (excluding __pycache__ and private)."""
    mods = []
    for info in pkgutil.walk_packages(revo.__path__, prefix="revo."):
        if info.name.count(".") == 1 and not info.name.endswith("__"):
            mods.append(info.name)
    return sorted(mods)


MODULES = _list_modules()

# Map: module → expected top-level symbols (function/class names)
CONFIRMED_SYMBOLS: dict[str, list[str]] = {
    "revo.metric_field_pinn": ["MetricFieldPINN", "pde_residual", "train_pinn", "curvature_metric"],
    "revo.tqft": ["TQFTConfig", "TopologicalProtector"],
    "revo.latency_monitor": ["measure_latency_distribution"],
    "revo.equilibrium_propagation": ["equilibrium_propagation_tune"],
    "revo.engine": ["DeltaHandle", "apply_delta", "revert_delta"],
    "revo.mode_cache": ["ModeCache"],
    "revo.pipeline": ["PhaseRunner", "run_pipeline_cli"],
    "revo.context_encoder": ["WindowContextEncoder", "EMAContextEncoder", "FourierContextEncoder"],
    "revo.ephemeral_engine": ["EphemeralConfig", "EphemeralEngine"],
}


def test_all_modules_import():
    """Every revo/*.py module imports without ImportError."""
    failed = []
    for mod_name in MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            failed.append((mod_name, str(e)))
    assert not failed, f"Import failures: {failed}"


def test_expected_symbols_present():
    """Key symbols exist in each module that has them listed."""
    for mod_name, symbols in CONFIRMED_SYMBOLS.items():
        if symbols is None:
            continue
        mod = importlib.import_module(mod_name)
        for sym in symbols:
            assert hasattr(mod, sym), f"{mod_name} missing {sym}"
            obj = getattr(mod, sym)
            assert inspect.isclass(obj) or inspect.isfunction(obj) or callable(obj), \
                f"{mod_name}.{sym} is not callable/class"
