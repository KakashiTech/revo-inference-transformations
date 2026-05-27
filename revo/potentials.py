from __future__ import annotations

import os
import json
import time
import base64
import uuid
from typing import Any, Dict, Optional

import numpy as np


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _rotate_if_needed(path: str, max_mb: float) -> None:
    try:
        if max_mb <= 0 or not os.path.exists(path):
            return
        size = os.path.getsize(path)
        if size <= max_mb * 1024 * 1024:
            return
        ts = f"{int(time.time() * 1_000_000)}_{uuid.uuid4().hex[:6]}"
        os.replace(path, f"{path}.{ts}.bak")
    except Exception as e:
        print(f"[WARN] potentials rotation failed: {e}")


def _q16_b64(ctx: np.ndarray) -> str:
    try:
        arr = ctx.astype(np.float16, copy=False).tobytes()
    except Exception:
        arr = ctx.astype(np.float32, copy=False).tobytes()
    return base64.b64encode(arr).decode("ascii")


def log_potential(
    mode_key: str,
    features: Dict[str, Any],
    ctx: np.ndarray,
    scale: float,
    log_path: Optional[str] = None,
    codebook_path: Optional[str] = None,
    include_codebook: bool = True,
) -> None:
    """Append a JSONL record to potentials and optional codebook.

    Env: REVO_POTENTIALS_LOG, REVO_CODEBOOK, REVO_LOG_MAX_MB
    """
    log_path = log_path or os.environ.get("REVO_POTENTIALS_LOG", os.path.join("logs", "potentials", "potentials.jsonl"))
    codebook_path = codebook_path or os.environ.get("REVO_CODEBOOK", os.path.join("logs", "potentials", "codebook.jsonl"))
    try:
        max_mb = float(os.environ.get("REVO_LOG_MAX_MB", "16") or 16.0)
    except Exception:
        max_mb = 16.0

    rec = {
        "ts": time.time(),
        "mode_key": str(mode_key or ""),
        "features": dict(features or {}),
        "ctx_mean_abs": float(np.mean(np.abs(ctx.astype(np.float32)))) if ctx is not None else 0.0,
        "ctx_dim": int(ctx.size if ctx is not None else 0),
        "scale": float(scale),
    }
    try:
        _ensure_dir(log_path)
        _rotate_if_needed(log_path, max_mb)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] potentials log write failed: {e}")

    if include_codebook and ctx is not None and ctx.size > 0:
        try:
            _ensure_dir(codebook_path)
            _rotate_if_needed(codebook_path, max_mb)
            entry = {
                "ts": time.time(),
                "mode_key": str(mode_key or ""),
                "ctx_q16_b64": _q16_b64(ctx),
                "ctx_dim": int(ctx.size),
                "version": 1,
            }
            with open(codebook_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] potentials codebook write failed: {e}")
