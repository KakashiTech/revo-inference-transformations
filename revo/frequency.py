from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import seed_everything, DEVICE


def _encode(tok, text: str, max_len: int) -> Dict[str, torch.Tensor]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(DEVICE) for k, v in enc.items()}


def _entropy_from_logits(logits: torch.Tensor) -> float:
    p = torch.log_softmax(logits, dim=-1).exp()
    h = -torch.sum(p * torch.log(p + 1e-12)).item()
    return float(h)


@dataclass
class FreqConfig:
    bands: Dict[str, int] = None  # cycles per sequence
    phases: int = 8  # number of phase steps in [0, 2pi)

    def __post_init__(self):
        if self.bands is None:
            # cycles per sequence for token index t in [0,1]
            self.bands = {
                "delta": 1,
                "theta": 2,
                "alpha": 4,
                "beta": 8,
                "gamma": 16,
            }


@dataclass
class FreqRun:
    name: str
    nll_general: float
    nll_ood: float
    mean_best_coherence: float
    band_hist: Dict[str, float]
    eval_time_s: float


def _load_model_tok(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(DEVICE)
    return model, tok


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
    ]
    return [base[i % len(base)] + f" [FREQ G#{i}]" for i in range(n)]


def _texts_ood(n: int) -> List[str]:
    base = [
        "Summarize the role of spectral sparsity in compression.",
        "Write a short Python function for radix prefix search.",
        "Explain energy vs Landauer limit in simple terms.",
        "What is reversible computing and why does it matter?",
        "Give an example of wave-division multiplexing in optics.",
    ]
    return [base[i % len(base)] + f" [FREQ O#{i}]" for i in range(n)]


def _nll(model, tok, texts: List[str], max_len: int) -> float:
    total, tokens = 0.0, 0
    with torch.no_grad():
        for t in texts:
            batch = _encode(tok, t, max_len)
            out = model(**batch, labels=batch["input_ids"])  # type: ignore[arg-type]
            loss = float(out.loss.item())
            n_tok = int(batch["input_ids"].numel())
            total += loss * n_tok
            tokens += n_tok
    return float(total / max(1, tokens))


def _sequence_best_band_and_coherence(logits_seq: torch.Tensor, cfg: FreqConfig) -> Tuple[str, float]:
    # logits_seq: [T, V]
    T = logits_seq.shape[0]
    if T < 3:
        return "delta", 0.0
    H = []
    with torch.no_grad():
        for t in range(T):
            H.append(_entropy_from_logits(logits_seq[t]))
    H = np.array(H, dtype=np.float64)
    # relevance ~ low entropy
    R = (H.max() - H)
    # standardize
    R = (R - R.mean()) / (R.std() + 1e-12)
    tnorm = np.linspace(0.0, 1.0, T, endpoint=False)
    best_band = None
    best_coh = -1.0
    for band, cycles in cfg.bands.items():
        # precompute waveforms for phase grid
        phis = np.linspace(0.0, 2.0 * math.pi, num=cfg.phases, endpoint=False)
        for phi in phis:
            s = np.sin(2.0 * math.pi * cycles * tnorm + float(phi))
            s = (s - s.mean()) / (s.std() + 1e-12)
            # Pearson correlation
            coh = float(np.dot(R, s) / (len(R) - 1))
            coh = abs(coh)
            if coh > best_coh:
                best_coh = coh
                best_band = band
    return best_band or "delta", float(best_coh)


def evaluate_frequency(
    model_name: str,
    seeds: List[int],
    prompts: int,
    max_len: int,
    cfg: FreqConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or FreqConfig()
    gen_texts = _texts_general(prompts)
    ood_texts = _texts_ood(max(50, prompts // 2))

    runs: List[FreqRun] = []
    model, tok = _load_model_tok(model_name)

    for seed in seeds:
        seed_everything(seed)
        t0 = time.perf_counter()
        nll_gen = _nll(model, tok, gen_texts, max_len)
        nll_ood = _nll(model, tok, ood_texts, max_len)
        bands = []
        cohs = []
        with torch.no_grad():
            for t in gen_texts:
                batch = _encode(tok, t, max_len)
                out = model(**batch)
                logits = out.logits.detach().float()[0]
                band, coh = _sequence_best_band_and_coherence(logits, cfg)
                bands.append(band); cohs.append(coh)
        elapsed = time.perf_counter() - t0
        hist: Dict[str, float] = {}
        for b in cfg.bands.keys():
            hist[b] = float(np.mean([1.0 if x == b else 0.0 for x in bands]))
        runs.append(FreqRun(
            name=f"freq_seed{seed}",
            nll_general=float(nll_gen),
            nll_ood=float(nll_ood),
            mean_best_coherence=float(np.mean(cohs) if cohs else 0.0),
            band_hist=hist,
            eval_time_s=elapsed,
        ))

    payload: Dict[str, object] = {
        "model": model_name,
        "seeds": seeds,
        "prompts": prompts,
        "max_length": max_len,
        "freq_config": {"bands": cfg.bands, "phases": cfg.phases},
        "variants": [asdict(r) for r in runs],
    }
    return payload


def run_frequency_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run Cognitive Frequency Tuning (Phase II.5)")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--phases", type=int, default=8)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cfg = FreqConfig()
    cfg.phases = int(args.phases)
    result = evaluate_frequency(args.model, seeds, int(args.prompts), int(args.max_length), cfg)

    out_path = args.results_json or os.path.join("quality", "phase2_runs", f"ii5_frequency_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results_path": out_path}, indent=2))


if __name__ == "__main__":
    run_frequency_cli()
