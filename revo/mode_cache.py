"""ModeCache con búsqueda por similitud de Z (coseno).

Extiende el ModeCache original con:
  - Almacenamiento del vector Z junto con cada entrada
  - Búsqueda por similitud coseno: get_similar(Z, threshold)
  - Compatible hacia atrás con get/put por key exacta
"""

from __future__ import annotations

import os
import json
import time
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 1e-10 else 0.0


class ModeCache:
    """Modo caché con TTL, LRU, y búsqueda por similitud Z.

    Env: REVO_MODECACHE_TTL_SECONDS, REVO_MODECACHE_MAX, REVO_MODES_LOG
    """

    def __init__(self,
                 ttl_seconds: Optional[int] = None,
                 max_size: Optional[int] = None,
                 log_path: Optional[str] = None,
                 similarity_threshold: float = 0.92):
        self.ttl = int(ttl_seconds if ttl_seconds is not None
                       else int(os.environ.get("REVO_MODECACHE_TTL_SECONDS", "900") or 900))
        self.max_size = int(max_size if max_size is not None
                            else int(os.environ.get("REVO_MODECACHE_MAX", "128") or 128))
        self.log_path = log_path or os.environ.get("REVO_MODES_LOG",
                                                   os.path.join("logs", "modes", "mode_cache.jsonl"))
        self.similarity_threshold = similarity_threshold
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._log_dir_created = False

    def _log_event(self, obj: Dict[str, Any]) -> None:
        try:
            with self._lock:
                d = os.path.dirname(self.log_path)
                if d and not self._log_dir_created:
                    os.makedirs(d, exist_ok=True)
                    self._log_dir_created = True
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _evict_if_needed(self) -> None:
        with self._lock:
            if len(self._cache) < int(self.max_size):
                return
            oldest_k = None
            oldest_ts = 9e99
            for k, v in self._cache.items():
                ts = float(v.get("ts", 0.0))
                if ts < oldest_ts:
                    oldest_ts = ts
                    oldest_k = k
            if oldest_k is not None:
                self._cache.pop(oldest_k, None)
        if oldest_k is not None:
            self._log_event({"ts": time.time(), "event": "evict_size", "key": str(oldest_k)})

    def get(self, key: str):
        """Get by exact key."""
        with self._lock:
            ent = self._cache.get(str(key) or "")
            if not ent:
                self._log_event({"ts": time.time(), "event": "exact_miss", "key": str(key)})
                return None
            ts = float(ent.get("ts", 0.0))
            if (time.time() - ts) > float(self.ttl):
                self._cache.pop(str(key), None)
                self._log_event({"ts": time.time(), "event": "evict_ttl", "key": str(key)})
                return None
            ent["ts"] = time.time()
        self._log_event({"ts": time.time(), "event": "exact_hit", "key": str(key)})
        return ent.get("ctx")

    def get_similar(self, z: np.ndarray) -> Tuple[Optional[Any], Optional[str], float]:
        """Find cached entry by cosine similarity to Z.

        Returns: (ctx, key, similarity). If none above threshold, all None.
        """
        best_key: Optional[str] = None
        best_sim = -1.0
        best_ent: Optional[Dict] = None

        with self._lock:
            now = time.time()
            for key, ent in list(self._cache.items()):
                ts = float(ent.get("ts", 0.0))
                if (now - ts) > float(self.ttl):
                    self._cache.pop(key, None)
                    self._log_event({"ts": now, "event": "evict_ttl", "key": key})
                    continue
                z_stored = ent.get("z")
                if z_stored is None:
                    continue
                sim = _cosine_sim(z, np.array(z_stored, dtype=np.float32))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
                    best_ent = ent

        if best_key is not None and best_sim >= self.similarity_threshold:
            best_ent["ts"] = time.time()
            self._log_event({"ts": time.time(), "event": "similar_hit",
                             "key": best_key, "similarity": round(best_sim, 4)})
            return best_ent.get("ctx"), best_key, best_sim

        self._log_event({"ts": time.time(), "event": "similar_miss",
                         "best_similarity": round(best_sim, 4) if best_sim >= 0 else -1.0})
        return None, None, best_sim if best_sim >= 0 else -1.0

    def put(self, key: str, ctx_any: Any,
            signature: Optional[Dict[str, Any]] = None,
            z: Optional[np.ndarray] = None) -> None:
        k = str(key) or ""
        if not k:
            return
        with self._lock:
            ev = "put_update" if k in self._cache else "put"
            entry = {"ctx": ctx_any, "ts": time.time(), "sig": dict(signature or {})}
            if z is not None:
                entry["z"] = z.tolist()
            self._cache[k] = entry
            self._evict_if_needed()
        self._log_event({"ts": time.time(), "event": ev, "key": k})

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"size": len(self._cache), "max_size": self.max_size, "ttl": self.ttl}
