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
    return [base[i % len(base)] + f" [BEDS G#{i}]" for i in range(n)]


def _texts_ood(n: int) -> List[str]:
    base = [
        "Summarize the role of spectral sparsity in compression.",
        "Write a short Python function for radix prefix search.",
        "Explain energy vs Landauer limit in simple terms.",
        "What is reversible computing and why does it matter?",
        "Give an example of wave-division multiplexing in optics.",
    ]
    return [base[i % len(base)] + f" [BEDS O#{i}]" for i in range(n)]


@dataclass
class BEDSConfig:
    target_entropy: float = 3.0
    ema_alpha: float = 0.1
    temp_min: float = 0.5
    temp_max: float = 2.0
    k_update: float = 0.1


@dataclass
class BEDSRun:
    name: str
    nll_general: float
    nll_ood: float
    entropy_homeostasis_mae: float
    exported_entropy: float
    eval_time_s: float


class BEDSController:
    def __init__(self, cfg: BEDSConfig):
        self.cfg = cfg
        self.H_hat = None  # type: float | None
        self.T = 1.0
        self.exported = 0.0

    def reset(self) -> None:
        self.H_hat = None
        self.T = 1.0
        self.exported = 0.0

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> float:
        p = torch.log_softmax(logits, dim=-1).exp()
        h = -torch.sum(p * torch.log(p + 1e-12)).item()
        return float(h)

    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        H = self._entropy_from_logits(logits)
        if self.H_hat is None:
            self.H_hat = H
        else:
            self.H_hat = (1.0 - self.cfg.ema_alpha) * self.H_hat + self.cfg.ema_alpha * H
        err = float(self.H_hat - self.cfg.target_entropy)
        # export "excess" entropy (positive error)
        if err > 0:
            self.exported += err
        # temperature update (multiplicative)
        self.T = float(np.clip(self.T * math.exp(self.cfg.k_update * err), self.cfg.temp_min, self.cfg.temp_max))
        return logits / max(1e-6, self.T)


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


def _nll_beds_sequence(model, tok, text: str, max_len: int, ctrl: BEDSController) -> Tuple[float, float, float]:
    with torch.no_grad():
        batch = _encode(tok, text, max_len)
        out = model(**batch)
        logits = out.logits.detach().float()[0]  # [T, V]
        ids = batch["input_ids"][0]  # [T]
        ctrl.reset()
        total = 0.0
        for t in range(logits.shape[0]):
            l = logits[t]
            ladj = ctrl.adjust_logits(l)
            logp = torch.log_softmax(ladj, dim=-1)
            total += -float(logp[ids[t]].item())
        nll = total / max(1, int(ids.numel()))
        H_mae = abs((ctrl.H_hat or 0.0) - ctrl.cfg.target_entropy)
        exported = ctrl.exported
        return float(nll), float(H_mae), float(exported)


def evaluate_beds(
    model_name: str,
    seeds: List[int],
    prompts: int,
    max_len: int,
    cfg: BEDSConfig | None = None,
) -> Dict[str, object]:
    cfg = cfg or BEDSConfig()
    gen_texts = _texts_general(prompts)
    ood_texts = _texts_ood(max(50, prompts // 2))

    metrics: List[BEDSRun] = []
    model, tok = _load_model_tok(model_name)

    for seed in seeds:
        seed_everything(seed)
        ctrl = BEDSController(cfg)
        t0 = time.perf_counter()
        nlls_g, maes_g, exp_g = [], [], []
        nlls_o, maes_o, exp_o = [], [], []
        for t in gen_texts:
            nll, mae, ex = _nll_beds_sequence(model, tok, t, max_len, ctrl)
            nlls_g.append(nll); maes_g.append(mae); exp_g.append(ex)
        for t in ood_texts:
            nll, mae, ex = _nll_beds_sequence(model, tok, t, max_len, ctrl)
            nlls_o.append(nll); maes_o.append(mae); exp_o.append(ex)
        elapsed = time.perf_counter() - t0
        metrics.append(BEDSRun(
            name=f"beds_seed{seed}",
            nll_general=float(np.mean(nlls_g)),
            nll_ood=float(np.mean(nlls_o)),
            entropy_homeostasis_mae=float(np.mean(maes_g + maes_o)),
            exported_entropy=float(np.mean(exp_g + exp_o)),
            eval_time_s=elapsed,
        ))

    payload: Dict[str, object] = {
        "model": model_name,
        "seeds": seeds,
        "prompts": prompts,
        "max_length": max_len,
        "beds_config": asdict(cfg),
        "variants": [asdict(m) for m in metrics],
    }
    return payload


def run_beds_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run BEDS (Phase II.1)")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--target-entropy", type=float, default=3.0)
    ap.add_argument("--ema-alpha", type=float, default=0.1)
    ap.add_argument("--temp-min", type=float, default=0.5)
    ap.add_argument("--temp-max", type=float, default=2.0)
    ap.add_argument("--k-update", type=float, default=0.1)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    cfg = BEDSConfig(
        target_entropy=float(args.target_entropy),
        ema_alpha=float(args.ema_alpha),
        temp_min=float(args.temp_min),
        temp_max=float(args.temp_max),
        k_update=float(args.k_update),
    )
    result = evaluate_beds(args.model, seeds, int(args.prompts), int(args.max_length), cfg)
    out_path = args.results_json or os.path.join("quality", "phase2_runs", f"ii1_beds_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results_path": out_path}, indent=2))


if __name__ == "__main__":
    run_beds_cli()
