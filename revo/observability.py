from __future__ import annotations

import os
import json
import time
import math
from typing import Optional
from collections import Counter


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def _read_last_ema(path: str):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            try:
                f.seek(-4096, os.SEEK_END)
            except OSError:
                f.seek(0, os.SEEK_SET)
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        for ln in reversed(lines):
            try:
                obj = json.loads(ln)
                ema = obj.get("anchor_ema")
                if isinstance(ema, dict):
                    return ema
            except Exception:
                continue
    except Exception:
        return None
    return None


def log_identity_anchor(text: str, model: str = "unknown", path: Optional[str] = None) -> None:
    """Append style/identity metrics to identity_anchor.jsonl."""
    try:
        out_path = path or os.path.join("logs", "revo", "identity_anchor.jsonl")
        _ensure_dir(out_path)
        txt = (text or "")
        import re
        sents = [z.strip() for z in re.split(r'[.!?]+', txt) if z.strip()]
        lens = [len(s.split()) for s in sents] or [0]
        avg_len = float(sum(lens)) / float(len(lens))
        var = float(sum((l - avg_len) ** 2 for l in lens)) / float(len(lens))
        std_len = float(var ** 0.5)
        words = [w.strip().lower() for w in txt.split() if w.strip()]
        V = max(1, len(set(words)))
        N = max(1, len(words))
        freqs = Counter(words)
        ent = 0.0
        for c in freqs.values():
            p = float(c) / float(N)
            if p > 0:
                ent -= p * math.log(p + 1e-12)
        ent_norm = float(ent / max(1e-9, math.log(float(V) + 1e-9))) if V > 1 else 0.0
        lines = [ln.strip().lower() for ln in txt.splitlines()]
        list_like = sum(1 for ln in lines if ln.startswith(("- ", "* ", "1.", "2.", "3.")))
        list_ratio = float(list_like) / float(max(1, len(lines)))
        bigrams = [(words[i], words[i+1]) for i in range(0, max(0, len(words) - 1))]
        B = len(bigrams)
        Ub = len(set(bigrams))
        bigram_cov = float(Ub) / float(max(1, B))
        try:
            alpha = float(os.environ.get("REVO_ID_ANCHOR_EMA_ALPHA", "0.4") or 0.4)
        except Exception:
            alpha = 0.4
        prev = _read_last_ema(out_path) or {}
        def _ema(k: str, now: float) -> float:
            prev_v = float(prev.get(k, now))
            return float(alpha * now + (1.0 - alpha) * prev_v)
        anchor_now = {
            "avg_sent_len": float(avg_len),
            "std_sent_len": float(std_len),
            "lex_entropy": float(max(0.0, min(1.0, ent_norm))),
            "list_ratio": float(max(0.0, min(1.0, list_ratio))),
            "bigram_cov": float(max(0.0, min(1.0, bigram_cov))),
        }
        anchor_ema = {
            "avg_sent_len": _ema("avg_sent_len", anchor_now["avg_sent_len"]),
            "std_sent_len": _ema("std_sent_len", anchor_now["std_sent_len"]),
            "lex_entropy": _ema("lex_entropy", anchor_now["lex_entropy"]),
            "list_ratio": _ema("list_ratio", anchor_now["list_ratio"]),
            "bigram_cov": _ema("bigram_cov", anchor_now["bigram_cov"]),
        }
        rec = {"ts": time.time(), "model": str(model), "anchor_now": anchor_now, "anchor_ema": anchor_ema}
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] identity_anchor write failed: {e}")


def log_cognitive_conservation(text: str, model: str = "unknown", path: Optional[str] = None) -> None:
    """Append semantic coverage proxy to cognitive_conservation.jsonl."""
    try:
        out_path = path or os.path.join("logs", "revo", "cognitive_conservation.jsonl")
        _ensure_dir(out_path)
        words = [w.strip().lower() for w in (text or "").split() if w.strip() and len(w) > 2 and w.isalpha()]
        unique_c = len(set(words))
        total_c = max(1, len(words))
        sem_cov = float(unique_c) / float(total_c)
        rec = {"ts": time.time(), "model": str(model), "semantic_coverage": float(sem_cov)}
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] cognitive_conservation write failed: {e}")
