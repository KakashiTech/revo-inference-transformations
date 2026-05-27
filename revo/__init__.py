"""REVO Compute: backend-agnostic reference (NumPy)."""
from .hyperlora import HyperLoraConfig, hyperlora_generate
from .engine import apply_delta, revert_delta, DeltaHandle
from .mode_cache import ModeCache
from .potentials import log_potential

__all__ = [
    "HyperLoraConfig",
    "hyperlora_generate",
    "apply_delta",
    "revert_delta",
    "DeltaHandle",
    "ModeCache",
    "log_potential",
]

__version__ = "0.1.0"
