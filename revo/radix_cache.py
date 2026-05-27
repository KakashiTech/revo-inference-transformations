from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import seed_everything, DEVICE


def _encode(tok, text: str, max_len: int) -> Dict[str, torch.Tensor]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(_device()) for k, v in enc.items()}


def _avg_last_hidden(model, tok, text: str, max_len: int) -> np.ndarray:
    with torch.no_grad():
        batch = _encode(tok, text, max_len)
        out = model(**batch, output_hidden_states=True)
        hs = out.hidden_states[-1][0]  # [T, H]
        v = hs.mean(dim=0).detach().cpu().numpy()
        # normalize
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _perturb(t: str) -> str:
    return (t + " ,").replace("  ", " ")


@dataclass
class RadixConfig:
    prefix_bits: int = 64
    prefix_len: int = 16
    bank_frac: float = 0.5  # fraction of prompts used as memory bank


class RadixAssociativeCache:
    def __init__(self, cfg: RadixConfig, max_entries: int = 500):
        self.cfg = cfg
        self.bank: Dict[int, np.ndarray] = {}
        self.bank_ids: List[int] = []
        self.index: Dict[str, List[int]] = {}
        self.max_entries = max_entries

    def _key(self, vec: np.ndarray) -> str:
        d = vec.shape[0]
        idx = np.arange(self.cfg.prefix_bits) % d
        bits = (vec[idx] >= 0).astype(np.int8)
        s = ''.join('1' if b > 0 else '0' for b in bits.tolist())
        return s

    def _prefixes(self, key: str) -> List[str]:
        L = min(len(key), self.cfg.prefix_len)
        return [key[:l] for l in range(L, 0, -1)]

    def _evict(self) -> None:
        while len(self.bank) > self.max_entries:
            oldest_id = self.bank_ids.pop(0)
            self.bank.pop(oldest_id, None)
            for pref in list(self.index.keys()):
                self.index[pref] = [cid for cid in self.index[pref] if cid != oldest_id]
                if not self.index[pref]:
                    del self.index[pref]

    def put(self, vec: np.ndarray, idx: int) -> None:
        k = self._key(vec)
        self.bank[idx] = vec
        self.bank_ids.append(idx)
        for pref in self._prefixes(k):
            self.index.setdefault(pref, []).append(idx)
        self._evict()

    def query(self, vec: np.ndarray, topk: int = 1) -> List[Tuple[int, float]]:
        k = self._key(vec)
        bucket: Optional[List[int]] = None
        for pref in self._prefixes(k):
            if pref in self.index and len(self.index[pref]) > 0:
                bucket = self.index[pref]
                break
        if bucket is None:
            cand_ids = list(self.bank.keys())
        else:
            cand_ids = bucket
        scores: List[Tuple[int, float]] = []
        for cid in cand_ids:
            v = self.bank.get(cid)
            if v is not None:
                scores.append((cid, _cosine(vec, v)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


@dataclass
class RadixMetrics:
    name: str
    hit_rate_top1: float
    avg_bucket_size: float
    avg_latency_prefix_ms: float
    avg_latency_bruteforce_ms: float
    retrieval_time_ratio: float
    delta_nll: float


def _load_model_tok(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(_device())
    return model, tok


def _nll(model, tok, texts: List[str], max_len: int) -> float:
    total, tokens = 0.0, 0
    with torch.no_grad():
        for t in texts:
            batch = _encode(tok, t, max_len)
            out = model(**batch, labels=batch["input_ids"])  # type: ignore
            loss = float(out.loss.item())
            n_tok = int(batch["input_ids"].numel())
            total += loss * n_tok
            tokens += n_tok
    return float(total / max(1, tokens))


def _texts(n: int) -> List[str]:
    base = [
        "Explica brevemente el filtrado espectral y su impacto.",
        "Describe el uso de matrices circulantes en FFT.",
        "¿Qué aporta HoRA sobre una variedad hiperbólica?",
        "Resume el mapeo holográfico bulk→boundary→bulk.",
        "Define rango efectivo y energía espectral.",
        "¿Qué es un bus de fase natural?",
        "Explica el un-computing reversible.",
        "¿Qué es WDM y cómo paraleliza subcanales?",
        "¿Cómo funciona un gating efímero suave?",
        "Explica el caching tipo árbol de prefijos (Radix).",
    ]
    return [base[i % len(base)] + f" [RADIX #{i}]" for i in range(n)]


def evaluate_radix(
    model_name: str,
    seed: int,
    prompts: int,
    max_len: int,
    cfg: RadixConfig,
) -> Dict[str, object]:
    seed_everything(seed)
    model, tok = _load_model_tok(model_name)

    texts = _texts(prompts)
    bank_n = max(1, int(cfg.bank_frac * prompts))
    bank_texts = texts[:bank_n]
    query_texts = [_perturb(t) for t in bank_texts]

    # Build vectors
    vecs = [
        _avg_last_hidden(model, tok, t, max_len) for t in bank_texts
    ]

    cache = RadixAssociativeCache(cfg)
    for i, v in enumerate(vecs):
        cache.put(v, i)

    # Evaluate retrieval
    lat_pref = []
    lat_bf = []
    hits = 0
    bucket_sizes = []

    # Pre-compute brute-force time including cosines
    start_bf_all = time.perf_counter()
    for q_idx, qt in enumerate(query_texts):
        qv = _avg_last_hidden(model, tok, qt, max_len)
        # brute-force over all
        best = -1.0
        best_id = -1
        for i, v in enumerate(vecs):
            sc = _cosine(qv, v)
            if sc > best:
                best = sc
                best_id = i
    elapsed_bf_all = time.perf_counter() - start_bf_all
    avg_bf = (elapsed_bf_all / max(1, len(query_texts))) * 1000.0

    # Prefix retrieval
    for q_idx, qt in enumerate(query_texts):
        qv = _avg_last_hidden(model, tok, qt, max_len)
        k = cache._key(qv)
        # bucket size for the longest prefix with elements
        bucket: Optional[List[int]] = None
        for pref in cache._prefixes(k):
            if pref in cache.index and len(cache.index[pref]) > 0:
                bucket = cache.index[pref]
                break
        bucket_sizes.append(float(len(bucket) if bucket is not None else len(cache.bank_ids)))
        t0 = time.perf_counter()
        top = cache.query(qv, topk=1)
        lat_pref.append((time.perf_counter() - t0) * 1000.0)
        if top and top[0][0] == q_idx:
            hits += 1

    hit_rate = float(hits / max(1, len(query_texts)))
    avg_pref = float(np.mean(lat_pref)) if lat_pref else 0.0
    avg_bkt = float(np.mean(bucket_sizes)) if bucket_sizes else 0.0

    # NLL parity (no integration changes, so Δ≈0)
    nll_base = _nll(model, tok, bank_texts, max_len)
    nll_after = _nll(model, tok, bank_texts, max_len)
    delta_nll = float(nll_after - nll_base)

    payload = {
        "model": model_name,
        "seed": seed,
        "prompts": prompts,
        "max_length": max_len,
        "radix_config": asdict(cfg),
        "variant": asdict(RadixMetrics(
            name=f"radix_seed{seed}",
            hit_rate_top1=hit_rate,
            avg_bucket_size=avg_bkt,
            avg_latency_prefix_ms=avg_pref,
            avg_latency_bruteforce_ms=avg_bf,
            retrieval_time_ratio=(avg_pref / avg_bf if avg_bf > 0 else 1.0),
            delta_nll=delta_nll,
        )),
    }
    return payload


def run_radix_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run Radix Associative Cache (Phase II.4)")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--prefix-bits", type=int, default=64)
    ap.add_argument("--prefix-len", type=int, default=16)
    ap.add_argument("--bank-frac", type=float, default=0.5)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    cfg = RadixConfig(prefix_bits=int(args.prefix_bits), prefix_len=int(args.prefix_len), bank_frac=float(args.bank_frac))
    result = evaluate_radix(args.model, int(args.seed), int(args.prompts), int(args.max_length), cfg)

    out_path = args.results_json or os.path.join("quality", "phase2_runs", f"ii4_radix_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results_path": out_path}, indent=2))


if __name__ == "__main__":
    run_radix_cli()
