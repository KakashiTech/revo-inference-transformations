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


@dataclass
class TQFTConfig:
    topk: int = 64
    num_braids: int = 5
    noise_sigma: float = 0.05
    seed: int = 0


class TopologicalProtector:
    """Topological protection via cyclic group actions in frequency domain.

    Replaces random-permutation "braids" with FFT-based phase rotations.
    Each "braid" is a distinct phase rotation in frequency space; the median
    across braids suppresses noise that is not invariant under cyclic shifts.
    """

    def __init__(self, cfg: TQFTConfig):
        self.cfg = cfg
        rng = np.random.RandomState(cfg.seed)
        n_freq = cfg.topk // 2 + 1
        self._phases: List[torch.Tensor] = [
            torch.from_numpy(rng.uniform(0.0, 2.0 * math.pi, size=n_freq).astype(np.float32))
            for _ in range(cfg.num_braids)
        ]

    def protect_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [V]
        logp = torch.log_softmax(logits, dim=-1)
        V = logp.shape[-1]
        K = min(self.cfg.topk, V)
        vals, idx = torch.topk(logp, K, dim=-1)

        stacked = []
        for phase in self._phases:
            fft_vals = torch.fft.rfft(vals, n=K)
            rotated = fft_vals * torch.exp(1j * phase.to(fft_vals.device))
            protected = torch.fft.irfft(rotated, n=K)
            stacked.append(protected)

        stacked_t = torch.stack(stacked, dim=0)
        agg_vals = torch.median(stacked_t, dim=0)[0]

        prob = torch.exp(logp).clone()
        prob[idx] = torch.exp(agg_vals).to(prob.device, dtype=prob.dtype)
        prob = prob / (prob.sum() + 1e-12)
        return torch.log(prob + 1e-12)

    def logical_error(self, logits: torch.Tensor) -> float:
        """Return variance across phase-rotated protections (lower = more stable)."""
        logp = torch.log_softmax(logits, dim=-1)
        V = logp.shape[-1]
        K = min(self.cfg.topk, V)
        vals, idx = torch.topk(logp, K, dim=-1)

        protected = []
        for phase in self._phases:
            fft_vals = torch.fft.rfft(vals, n=K)
            rotated = fft_vals * torch.exp(1j * phase.to(fft_vals.device))
            pvals = torch.fft.irfft(rotated, n=K)
            protected.append(pvals)

        stacked_t = torch.stack(protected, dim=0)
        return float(torch.var(stacked_t, dim=0).mean().item())


def _load_model_tok(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(_device())
    return model, tok


def _encode(tok, text: str, max_len: int) -> Dict[str, torch.Tensor]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(_device()) for k, v in enc.items()}


def _nll_tqft_sequence(model, tok, text: str, max_len: int, protector: TopologicalProtector) -> Tuple[float, float]:
    with torch.no_grad():
        batch = _encode(tok, text, max_len)
        out = model(**batch)
        logits = out.logits.detach().float()[0]  # [T, V]
        ids = batch["input_ids"][0]  # [T]
        total = 0.0
        errors = []
        for t in range(logits.shape[0]):
            lp = protector.protect_logprobs(logits[t])
            total += -float(lp[ids[t]].item())
            errors.append(protector.logical_error(logits[t]))
        nll = total / max(1, int(ids.numel()))
        err_rate = float(np.mean(errors)) if errors else 0.0
        return float(nll), err_rate


def evaluate_tqft(
    model_name: str,
    seeds: List[int],
    prompts: int,
    max_len: int,
    cfg: TQFTConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or TQFTConfig()

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
        return [base[i % len(base)] + f" [TQFT G#{i}]" for i in range(n)]

    def _texts_ood(n: int) -> List[str]:
        base = [
            "Summarize the role of spectral sparsity in compression.",
            "Write a short Python function for radix prefix search.",
            "Explain energy vs Landauer limit in simple terms.",
            "What is reversible computing and why does it matter?",
            "Give an example of wave-division multiplexing in optics.",
        ]
        return [base[i % len(base)] + f" [TQFT O#{i}]" for i in range(n)]

    gen_texts = _texts_general(prompts)
    ood_texts = _texts_ood(max(50, prompts // 2))

    variants = []
    model, tok = _load_model_tok(model_name)
    for seed in seeds:
        seed_everything(seed)
        protector = TopologicalProtector(TQFTConfig(**asdict(cfg)))
        t0 = time.perf_counter()
        nlls_g, errs_g = [], []
        nlls_o, errs_o = [], []
        for t in gen_texts:
            nll, err = _nll_tqft_sequence(model, tok, t, max_len, protector)
            nlls_g.append(nll); errs_g.append(err)
        for t in ood_texts:
            nll, err = _nll_tqft_sequence(model, tok, t, max_len, protector)
            nlls_o.append(nll); errs_o.append(err)
        elapsed = time.perf_counter() - t0
        variants.append({
            "name": f"tqft_seed{seed}",
            "nll_general": float(np.mean(nlls_g)),
            "nll_ood": float(np.mean(nlls_o)),
            "logical_error_rate": float(np.mean(errs_g + errs_o)),
            "eval_time_s": elapsed,
        })

    return {
        "model": model_name,
        "seeds": seeds,
        "prompts": prompts,
        "max_length": max_len,
        "tqft_config": asdict(cfg),
        "variants": variants,
    }


def run_tqft_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run TQFT protection (Phase II.2)")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--num-braids", type=int, default=5)
    ap.add_argument("--noise-sigma", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cfg = TQFTConfig(topk=int(args.topk), num_braids=int(args.num_braids), noise_sigma=float(args.noise_sigma), seed=int(args.seed))
    result = evaluate_tqft(args.model, seeds, int(args.prompts), int(args.max_length), cfg)

    out_path = args.results_json or os.path.join("quality", "phase2_runs", f"ii2_tqft_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results_path": out_path}, indent=2))


if __name__ == "__main__":
    run_tqft_cli()
