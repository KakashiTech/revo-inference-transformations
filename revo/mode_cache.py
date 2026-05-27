from __future__ import annotations
from revo._logging import get_logger

import os
import json
import time
import threading
from typing import Any, Dict, Optional


class ModeCache:
    """Mode cache with TTL and LRU; logs events to JSONL.

    Env: REVO_MODECACHE_TTL_SECONDS, REVO_MODECACHE_MAX, REVO_MODES_LOG
    """

    def __init__(self,
                 ttl_seconds: Optional[int] = None,
                 max_size: Optional[int] = None,
                 log_path: Optional[str] = None) -> None:
        try:
            self.ttl = int(ttl_seconds if ttl_seconds is not None else os.environ.get("REVO_MODECACHE_TTL_SECONDS", "900") or 900)
        except Exception:
            self.ttl = 900
        try:
            self.max_size = int(max_size if max_size is not None else os.environ.get("REVO_MODECACHE_MAX", "64") or 64)
        except Exception:
            self.max_size = 64
        self.log_path = log_path or os.environ.get("REVO_MODES_LOG", os.path.join("logs", "modes", "mode_cache.jsonl"))
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
            get_logger().warning("except Exception:")
    def _evict_if_needed(self) -> None:
        with self._lock:
            if len(self._cache) < int(self.max_size):
                return
            oldest_k = None
            oldest_ts = 9e99
            for k, v in self._cache.items():
                try:
                    ts = float(v.get("ts", 0.0))
                except Exception:
                    ts = time.time()
                if ts < oldest_ts:
                    oldest_ts = ts
                    oldest_k = k
            if oldest_k is not None:
                self._cache.pop(oldest_k, None)
        if oldest_k is not None:
            self._log_event({"ts": time.time(), "event": "evict_size", "key": str(oldest_k)})

    def get(self, key: str):
        with self._lock:
            ent = self._cache.get(str(key) or "")
            if not ent:
                self._log_event({"ts": time.time(), "event": "exact_miss", "key": str(key)})
                return None
            try:
                ts = float(ent.get("ts", 0.0))
            except Exception:
                ts = time.time()
            if (time.time() - ts) > float(self.ttl):
                self._cache.pop(str(key), None)
                self._log_event({"ts": time.time(), "event": "evict_ttl", "key": str(key)})
                return None
            ent["ts"] = time.time()
        self._log_event({"ts": time.time(), "event": "exact_hit", "key": str(key)})
        return ent.get("ctx")

    def put(self, key: str, ctx_any: Any, signature: Optional[Dict[str, Any]] = None) -> None:
        k = str(key) or ""
        if not k:
            return
        with self._lock:
            ev = "put_update" if k in self._cache else "put"
            self._cache[k] = {"ctx": ctx_any, "ts": time.time(), "sig": dict(signature or {})}
            self._evict_if_needed()
        self._log_event({"ts": time.time(), "event": ev, "key": k})
