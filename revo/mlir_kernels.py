from __future__ import annotations

from typing import Optional

import torch


def compile_model_guarded(model: torch.nn.Module, backend: str = "inductor") -> torch.nn.Module:
    """Try to compile the model with torch.compile; fallback to original model on failure.

    backend options typically include 'inductor'. If torch.compile is unavailable, returns model.
    """
    try:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return model
        compiled = compile_fn(model, backend=backend, mode="max-autotune")
        return compiled
    except Exception:
        return model
