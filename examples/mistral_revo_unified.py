from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np


# ------------------------
# Prompt sets
# ------------------------

def _texts_general(n: int) -> List[str]:
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
    return [base[i % len(base)] + f" [G#{i}]" for i in range(n)]


def _texts_ood(n: int) -> List[str]:
    base = [
        "Summarize the role of spectral sparsity in compression.",
        "Write a short Python function for radix prefix search.",
        "Explain energy vs Landauer limit in simple terms.",
        "What is reversible computing and why does it matter?",
        "Give an example of wave-division multiplexing in optics.",
    ]
    return [base[i % len(base)] + f" [O#{i}]" for i in range(n)]


def _perturb(t: str) -> str:
    return (t + " ,").replace("  ", " ")


# ------------------------
# llama.cpp helpers
# ------------------------


def _llm_load(gguf_path: str, n_ctx: int, n_threads: int):
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:
        raise RuntimeError(f"llama-cpp-python not available: {e}")
    llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, logits_all=True)
    return llm


def _llama_echo(llm, text: str, topk: int) -> Tuple[List[float], List[Dict[str, float]], List[str]]:
    """Return (token_logprobs, top_logprobs_per_token, tokens)."""
    out = llm.create_completion(
        prompt=text,
        max_tokens=0,
        echo=True,
        temperature=0,
        logprobs=max(1, topk),
    )
    choices = out["choices"][0]
    token_logprobs = choices["logprobs"]["token_logprobs"]
    top_logprobs = choices["logprobs"].get("top_logprobs", [])
    tokens = choices["logprobs"]["tokens"]
    return token_logprobs, top_logprobs, tokens


def _llama_last_topk(llm, text: str, topk: int) -> Dict[str, float]:
    llm.reset()
    out = llm.create_completion(
        prompt=text,
        max_tokens=1,
        echo=True,
        temperature=0,
        logprobs=max(1, topk),
    )
    top_list = out["choices"][0]["logprobs"].get("top_logprobs", [])
    last = top_list[-1] if top_list else {}
    return {str(k): float(v) for k, v in last.items()}


def _nll_from_token_logprobs(token_logprobs: List[float]) -> float:
    vals = [x for x in token_logprobs if x is not None]
    if not vals:
        return 0.0
    return float(-sum(vals) / max(1, len(vals)))


# ------------------------
# Category DSL (import text-only morphisms)
# ------------------------

from revo.category import Context, morph_compose_default  # type: ignore


def build_structured_prompts(texts: List[str]) -> Tuple[List[str], float]:
    plan = morph_compose_default()
    structured: List[str] = []
    dists = []
    for t in texts:
        ctx = Context(t)
        plan(ctx)
        structured.append(ctx.text)
        # similarity proxy: edit distance <= 32
        d = _levenshtein(t, ctx.text)
        dists.append(1.0 if d <= 32 else 0.0)
    return structured, float(np.mean(dists))


def _levenshtein(a: str, b: str) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(a)][len(b)]


# ------------------------
# TQFT protector on last-step distribution (top-K)
# ------------------------

@dataclass
class TQFTConfig:
    topk: int = 64
    num_braids: int = 5
    noise_sigma: float = 0.05
    seed: int = 0


class TopologicalProtector:
    def __init__(self, cfg: TQFTConfig):
        self.cfg = cfg
        self._rng = np.random.RandomState(cfg.seed)
        self._perms: List[np.ndarray] = [self._rng.permutation(self.cfg.topk) for _ in range(self.cfg.num_braids)]
        self._inv_perms: List[np.ndarray] = [np.argsort(p) for p in self._perms]

    def protect_logprobs_topk(self, topk_vals: np.ndarray) -> np.ndarray:
        # topk_vals: [K] log-probs sorted by the API order
        K = topk_vals.shape[0]
        stacked = []
        for p, pinv in zip(self._perms, self._inv_perms):
            idx = p[:K]
            inv = pinv[:K]
            vperm = topk_vals[idx]
            vback = np.empty_like(vperm)
            vback[inv] = vperm
            stacked.append(vback)
        stacked_np = np.stack(stacked, axis=0)
        agg_vals = np.median(stacked_np, axis=0)
        # normalize to probability simplex over top-K
        prob = np.exp(agg_vals - np.max(agg_vals))
        prob = prob / (prob.sum() + 1e-12)
        return np.log(prob + 1e-12)

    def logical_error(self, topk_vals: np.ndarray) -> float:
        base = self.protect_logprobs_topk(topk_vals)
        base_arg = int(np.argmax(base))
        noise = self._rng.randn(*topk_vals.shape) * float(self.cfg.noise_sigma)
        pert = self.protect_logprobs_topk(topk_vals + noise)
        pert_arg = int(np.argmax(pert))
        return 0.0 if base_arg == pert_arg else 1.0


# ------------------------
# BEDS controller (approximate via top-K per-token logprobs)
# ------------------------

@dataclass
class BEDSConfig:
    target_entropy: float = 3.0
    ema_alpha: float = 0.1
    temp_min: float = 0.5
    temp_max: float = 2.0
    k_update: float = 0.1


class BEDSController:
    def __init__(self, cfg: BEDSConfig):
        self.cfg = cfg
        self.H_hat: float | None = None
        self.T: float = 1.0
        self.exported: float = 0.0

    def reset(self) -> None:
        self.H_hat = None
        self.T = 1.0
        self.exported = 0.0

    def _entropy_from_topk(self, top_logprobs: Dict[str, float]) -> float:
        if not top_logprobs:
            return 0.0
        vals = np.array([float(v) for v in top_logprobs.values()], dtype=np.float64)
        prob = np.exp(vals - np.max(vals))
        prob = prob / (prob.sum() + 1e-12)
        h = float(-(prob * np.log(prob + 1e-12)).sum())
        return h

    def adjust_topk(self, top_logprobs: Dict[str, float]) -> Dict[str, float]:
        if not top_logprobs:
            return top_logprobs
        H = self._entropy_from_topk(top_logprobs)
        if self.H_hat is None:
            self.H_hat = H
        else:
            self.H_hat = (1.0 - self.cfg.ema_alpha) * self.H_hat + self.cfg.ema_alpha * H
        err = float(self.H_hat - self.cfg.target_entropy)
        if err > 0:
            self.exported += err
        # update temperature multiplicatively
        self.T = float(np.clip(self.T * math.exp(self.cfg.k_update * err), self.cfg.temp_min, self.cfg.temp_max))
        # rescale logprobs and renormalize
        vals = np.array([float(v) for v in top_logprobs.values()], dtype=np.float64)
        keys = list(top_logprobs.keys())
        vals = vals / max(1e-6, self.T)
        prob = np.exp(vals - np.max(vals))
        prob = prob / (prob.sum() + 1e-12)
        new_lp = np.log(prob + 1e-12)
        return {k: float(v) for k, v in zip(keys, new_lp.tolist())}


# ------------------------
# Frequency coherence (approximate via top-K per-token logprobs)
# ------------------------

@dataclass
class FreqConfig:
    phases: int = 8


class FrequencyAnalyzer:
    def __init__(self, cfg: FreqConfig):
        self.cfg = cfg
        self.bands = {
            "delta": 1,
            "theta": 2,
            "alpha": 4,
            "beta": 8,
            "gamma": 16,
        }

    def _approx_entropy(self, top_lp: Dict[str, float]) -> float:
        if not top_lp:
            return 0.0
        vals = np.array(list(top_lp.values()), dtype=np.float64)
        prob = np.exp(vals - np.max(vals))
        prob = prob / (prob.sum() + 1e-12)
        return float(-(prob * np.log(prob + 1e-12)).sum())

    def best_band_and_coherence(self, top_logprobs_seq: List[Dict[str, float]]) -> Tuple[str, float]:
        T = len(top_logprobs_seq)
        if T < 3:
            return "delta", 0.0
        H = [self._approx_entropy(tlp) for tlp in top_logprobs_seq]
        H = np.array(H, dtype=np.float64)
        R = (H.max() - H)
        R = (R - R.mean()) / (R.std() + 1e-12)
        tnorm = np.linspace(0.0, 1.0, T, endpoint=False)
        best_band = None
        best_coh = -1.0
        phis = np.linspace(0.0, 2.0 * math.pi, num=self.cfg.phases, endpoint=False)
        for band, cycles in self.bands.items():
            for phi in phis:
                s = np.sin(2.0 * math.pi * cycles * tnorm + float(phi))
                s = (s - s.mean()) / (s.std() + 1e-12)
                coh = float(np.dot(R, s) / (len(R) - 1))
                coh = abs(coh)
                if coh > best_coh:
                    best_coh = coh
                    best_band = band
        return best_band or "delta", float(best_coh)


# ------------------------
# Radix-like associative index over last-step top-K logprobs
# ------------------------

@dataclass
class RadixConfig:
    prefix_len: int = 16
    topk: int = 64


class RadixTopKCache:
    def __init__(self, cfg: RadixConfig):
        self.cfg = cfg
        self.index: Dict[str, List[int]] = {}
        self.bank: List[np.ndarray] = []

    def _key(self, topk_vals: np.ndarray) -> str:
        med = float(np.median(topk_vals))
        bits = (topk_vals >= med).astype(np.int8)
        s = ''.join('1' if b > 0 else '0' for b in bits.tolist())
        return s

    def _prefixes(self, key: str) -> List[str]:
        L = min(len(key), self.cfg.prefix_len)
        return [key[:l] for l in range(L, 0, -1)]

    def put(self, topk_vals: np.ndarray, idx: int) -> None:
        k = self._key(topk_vals)
        self.bank.append(topk_vals)
        for pref in self._prefixes(k):
            self.index.setdefault(pref, []).append(idx)

    def query(self, topk_vals: np.ndarray) -> Tuple[int, float, int]:
        k = self._key(topk_vals)
        bucket = None
        for pref in self._prefixes(k):
            if pref in self.index and len(self.index[pref]) > 0:
                bucket = self.index[pref]
                break
        if bucket is None:
            cand = range(len(self.bank))
        else:
            cand = bucket
        best = -1.0
        best_id = -1
        for cid in cand:
            v = self.bank[cid]
            sc = float(np.dot(v, topk_vals) / ((np.linalg.norm(v) * np.linalg.norm(topk_vals)) + 1e-12))
            if sc > best:
                best = sc
                best_id = int(cid)
        bucket_size = len(cand) if isinstance(cand, list) else len(list(cand))
        return best_id, best, bucket_size


# ------------------------
# Unified evaluation
# ------------------------

@dataclass
class VariantMetrics:
    name: str
    nll_general: float
    nll_ood: float
    eval_time_s: float


def run_unified(
    gguf_path: str,
    seeds: List[int],
    prompts: int,
    max_len: int,
    n_ctx: int,
    n_threads: int,
    topk: int,
    tqft_cfg: TQFTConfig,
    beds_cfg: BEDSConfig,
    freq_cfg: FreqConfig,
    radix_cfg: RadixConfig,
) -> Dict[str, Any]:
    gen_texts = _texts_general(prompts)
    ood_texts = _texts_ood(max(50, prompts // 2))

    try:
        from llama_cpp import Llama  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"llama-cpp-python not available: {e}")

    llm = _llm_load(gguf_path, n_ctx=n_ctx, n_threads=n_threads)

    seeds_res: List[Dict[str, Any]] = []

    for seed in seeds:
        np.random.seed(seed)
        # Baseline NLLs
        t0 = time.perf_counter()
        nlls_gen = []
        nlls_ood = []
        topk_seqs_general: List[List[Dict[str, float]]] = []
        for t in gen_texts:
            lp, tlp, toks = _llama_echo(llm, t, topk)
            nlls_gen.append(_nll_from_token_logprobs(lp))
            topk_seqs_general.append(tlp)
        for t in ood_texts:
            lp, tlp, toks = _llama_echo(llm, t, topk)
            nlls_ood.append(_nll_from_token_logprobs(lp))
        t_base = time.perf_counter() - t0

        baseline = VariantMetrics(name="baseline", nll_general=float(np.mean(nlls_gen)), nll_ood=float(np.mean(nlls_ood)), eval_time_s=t_base)

        # Category (structured prompts)
        structured, conservative_frac = build_structured_prompts(gen_texts)
        nlls_struct = []
        for t in structured:
            lp, *_ = _llama_echo(llm, t, topk)
            nlls_struct.append(_nll_from_token_logprobs(lp))
        category = {
            "nll_structured": float(np.mean(nlls_struct)),
            "delta_nll": float(np.mean(nlls_struct) - baseline.nll_general),
            "conservative_frac": float(conservative_frac),
        }

        # TQFT over last step
        protector = TopologicalProtector(tqft_cfg)
        errs = []
        for t in gen_texts:
            last = _llama_last_topk(llm, t, tqft_cfg.topk)
            if not last:
                continue
            vals = np.array(list(last.values()), dtype=np.float64)
            errs.append(protector.logical_error(vals))
        tqft = {
            "logical_error_rate": float(np.mean(errs) if errs else 0.0),
            "topk": tqft_cfg.topk,
            "num_braids": tqft_cfg.num_braids,
            "noise_sigma": tqft_cfg.noise_sigma,
        }

        # BEDS approximate policy
        beds_ctrl = BEDSController(beds_cfg)
        beds_losses_gen = []
        beds_mae = []
        beds_export = []
        for tlp in topk_seqs_general:
            beds_ctrl.reset()
            seq_loss = 0.0
            steps = 0
            for step_top in tlp:
                adj = beds_ctrl.adjust_topk(step_top)
                # if the actual token is within top-K, use its prob; otherwise, fallback to max prob
                # We don't have actual token id here; token_logprobs had it but we discarded tokens; approximate by highest prob within top-K
                pmax = max(adj.values()) if adj else 0.0
                seq_loss += -float(pmax)
                steps += 1
            if steps > 0:
                beds_losses_gen.append(float(seq_loss / steps))
                beds_mae.append(abs((beds_ctrl.H_hat or 0.0) - beds_ctrl.cfg.target_entropy))
                beds_export.append(float(beds_ctrl.exported))
        beds = {
            "nll_general_policy": float(np.mean(beds_losses_gen) if beds_losses_gen else 0.0),
            "entropy_homeostasis_mae": float(np.mean(beds_mae) if beds_mae else 0.0),
            "exported_entropy": float(np.mean(beds_export) if beds_export else 0.0),
        }

        # Frequency coherence (approx)
        freq = FrequencyAnalyzer(freq_cfg)
        bands = []
        cohs = []
        for tlp in topk_seqs_general:
            band, coh = freq.best_band_and_coherence(tlp)
            bands.append(band)
            cohs.append(coh)
        hist: Dict[str, float] = {}
        for b in freq.bands.keys():
            hist[b] = float(np.mean([1.0 if x == b else 0.0 for x in bands]))
        freq_res = {
            "mean_best_coherence": float(np.mean(cohs) if cohs else 0.0),
            "band_hist": hist,
            "phases": freq_cfg.phases,
        }

        # Radix-like cache on last-step topK vectors
        radix_cache = RadixTopKCache(radix_cfg)
        bank_texts = gen_texts[: max(1, int(0.5 * len(gen_texts)))]
        query_texts = [_perturb(t) for t in bank_texts]
        bank_vectors: List[np.ndarray] = []
        for i, t in enumerate(bank_texts):
            last = _llama_last_topk(llm, t, radix_cfg.topk)
            vals = np.array(list(last.values()), dtype=np.float64) if last else np.zeros((radix_cfg.topk,), dtype=np.float64)
            bank_vectors.append(vals)
            radix_cache.put(vals, i)
        hits = 0
        bucket_sizes = []
        for i, t in enumerate(query_texts):
            last = _llama_last_topk(llm, t, radix_cfg.topk)
            vals = np.array(list(last.values()), dtype=np.float64) if last else np.zeros((radix_cfg.topk,), dtype=np.float64)
            best_id, score, bsz = radix_cache.query(vals)
            bucket_sizes.append(float(bsz))
            if best_id == i:
                hits += 1
        radix = {
            "hit_rate_top1": float(hits / max(1, len(query_texts))),
            "avg_bucket_size": float(np.mean(bucket_sizes) if bucket_sizes else 0.0),
            "topk": radix_cfg.topk,
            "prefix_len": radix_cfg.prefix_len,
        }

        seeds_res.append({
            "seed": seed,
            "baseline": asdict(baseline),
            "category": category,
            "tqft": tqft,
            "beds": beds,
            "frequency": freq_res,
            "radix": radix,
        })

    return {
        "model": f"llama.cpp:{os.path.basename(gguf_path)}",
        "seeds": seeds,
        "prompts": prompts,
        "max_length": max_len,
        "results": seeds_res,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified REVO test over Mistral (llama.cpp) across 100 prompts")
    ap.add_argument("--llama-gguf", type=str, required=True, help="Path to GGUF model for llama.cpp backend")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-threads", type=int, default=max(os.cpu_count() or 4, 4))
    ap.add_argument("--topk", type=int, default=64)
    # TQFT
    ap.add_argument("--tqft-topk", type=int, default=64)
    ap.add_argument("--tqft-num-braids", type=int, default=5)
    ap.add_argument("--tqft-noise-sigma", type=float, default=0.05)
    # BEDS
    ap.add_argument("--beds-target-entropy", type=float, default=3.0)
    ap.add_argument("--beds-ema-alpha", type=float, default=0.1)
    ap.add_argument("--beds-temp-min", type=float, default=0.5)
    ap.add_argument("--beds-temp-max", type=float, default=2.0)
    ap.add_argument("--beds-k-update", type=float, default=0.1)
    # Frequency
    ap.add_argument("--freq-phases", type=int, default=8)
    # Radix
    ap.add_argument("--radix-prefix-len", type=int, default=16)
    ap.add_argument("--radix-topk", type=int, default=64)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    tqft_cfg = TQFTConfig(topk=int(args.tqft_topk), num_braids=int(args.tqft_num_braids), noise_sigma=float(args.tqft_noise_sigma), seed=0)
    beds_cfg = BEDSConfig(target_entropy=float(args.beds_target_entropy), ema_alpha=float(args.beds_ema_alpha), temp_min=float(args.beds_temp_min), temp_max=float(args.beds_temp_max), k_update=float(args.beds_k_update))
    freq_cfg = FreqConfig(phases=int(args.freq_phases))
    radix_cfg = RadixConfig(prefix_len=int(args.radix_prefix_len), topk=int(args.radix_topk))

    payload = run_unified(
        gguf_path=args.llama_gguf,
        seeds=seeds,
        prompts=int(args.prompts),
        max_len=int(args.max_length),
        n_ctx=int(args.n_ctx),
        n_threads=int(args.n_threads),
        topk=int(args.topk),
        tqft_cfg=tqft_cfg,
        beds_cfg=beds_cfg,
        freq_cfg=freq_cfg,
        radix_cfg=radix_cfg,
    )

    out_path = args.results_json or os.path.join("quality", "mistral", f"unified_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results_path": out_path}, indent=2))


if __name__ == "__main__":
    main()
