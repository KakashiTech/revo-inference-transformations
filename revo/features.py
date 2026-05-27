from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Dict, Optional

import numpy as np


def build_features(prompt: str) -> Dict[str, Any]:
    text = (prompt or "").lower()
    feats: Dict[str, Any] = {
        "prompt_len": len(prompt or ""),
        "has_code": any(k in text for k in ("def ", "class ", "python", "traceback")),
        "has_math": any(k in text for k in ("∑", "∫", "lim", "theorem", "proof")),
        "has_list": any(text.strip().startswith(p) for p in ("- ", "* ", "1.", "2.", "3.")),
        "lang_es": any(k in text for k in ("hola", "gracias", "¿", "ó", "ñ")),
        "lang_en": any(k in text for k in ("hello", "thanks", "the", "and")),
    }
    return feats


def compute_mode_key(feats: Dict[str, Any], prompt: str, take: int = 24) -> str:
    try:
        head = unicodedata.normalize('NFC', (prompt or "").strip().lower()[:160])
    except Exception:
        head = ""
    try:
        blob = json.dumps({"feats": feats, "head": head}, sort_keys=True, ensure_ascii=True)
    except Exception:
        blob = str(feats) + "|" + head
    h = hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()
    return h[:max(8, int(take))]


def build_context_vector(feats: Dict[str, Any], context_dim: int = 64, seed: int = 0) -> np.ndarray:
    """Deterministic context vector from features (sha256-based RNG)."""
    try:
        blob = json.dumps(feats, sort_keys=True, ensure_ascii=True)
    except Exception:
        blob = unicodedata.normalize('NFC', str(feats))
    h = hashlib.sha256(blob.encode("utf-8", errors="ignore")).digest()
    base = int.from_bytes(h[:8], "little") ^ int.from_bytes(h[8:16], "little") ^ int(seed or 0)
    rng = np.random.default_rng(base & 0x7FFFFFFF)
    ctx = rng.standard_normal(size=(int(context_dim),)).astype(np.float32) * 0.3
    return ctx
