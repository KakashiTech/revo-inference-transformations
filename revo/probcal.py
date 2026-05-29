from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import seed_everything, DEVICE


def _make_prompts(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    base = [
        "Define entropy and cross-entropy with examples.",
        "Explain the central limit theorem intuitively.",
        "Translate to French: Knowledge before expression.",
        "Write a sonnet about time dilation and love.",
        "List three use-cases for transformers in NLP.",
        "Derive the gradient of softmax cross-entropy.",
        "Summarize a research paper in three bullet points.",
        "Draft a memo arguing for reproducible ML.",
        "What are the risks of dataset shift?",
        "Compare L2 vs KL regularization in probabilistic terms.",
    ]
    prompts = []
    for i in range(n):
        s = base[i % len(base)]
        if rng.random() < 0.5:
            s += f" [seed={seed} v={rng.randint(0,9999)}]"
        prompts.append(s)
    return prompts


@dataclass
class ProbCalConfig:
    calib_frac: float = 0.5
    steps: int = 50
    lr: float = 0.05
    alpha_curv: float = 0.1
    topk_eval: int = 64


@torch.no_grad()
def _nll_and_entropy(model, tok, texts: List[str], max_length: int, topk_eval: int, tau: float | None = None, alpha_curv: float = 0.0) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    entropies: List[float] = []

    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(DEVICE)
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        logits = out.logits  # [1, T, V]
        if tau is not None:
            # local curvature via per-token entropy factor
            with torch.no_grad():
                base_probs = F.softmax(logits, dim=-1)
                token_H = -(base_probs * base_probs.clamp_min(1e-12).log()).sum(-1)  # [1, T]
                Href = float(token_H.mean()) + 1e-8
                curv = 1.0 + alpha_curv * (token_H / Href - 1.0)
            logits = logits / torch.clamp(curv, 0.5, 2.0).unsqueeze(-1) / max(1e-5, tau)
        probs = F.softmax(logits[:, :, :topk_eval], dim=-1)
        ent = float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean().item())
        entropies.append(ent)
        loss = float(out.loss.item()) if tau is None else float(_tokenwise_nll(input_ids, logits))
        n_tok = int(input_ids.numel())
        total_loss += loss * n_tok
        total_tokens += n_tok
    return float(total_loss / max(1, total_tokens)), float(sum(entropies) / max(1, len(entropies)))


def _tokenwise_nll(target_ids: torch.Tensor, logits: torch.Tensor) -> float:
    shift_logits = logits[:, :-1, :]
    shift_labels = target_ids[:, 1:]
    loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100)
    return float(loss.item())


def _make_ood(prompts: List[str]) -> List[str]:
    out = []
    for p in prompts:
        toks = p.split()
        toks = list(reversed(toks))
        out.append(" ".join(toks) + " [OOD]")
    return out


def _fit_temperature(model, tok, texts: List[str], max_length: int, steps: int, lr: float) -> float:
    tau = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([tau], lr=lr)
    device = DEVICE
    for _ in range(steps):
        opt.zero_grad()
        losses = []
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            logits = out.logits / torch.clamp(tau, 1e-3, 100.0)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1), ignore_index=-100)
            losses.append(loss)
        L = torch.stack(losses).mean()
        L.backward()
        opt.step()
        with torch.no_grad():
            tau.clamp_(1e-2, 100.0)
    return float(tau.detach().cpu().item())


def evaluate_probcal(
    model_name: str = "sshleifer/tiny-gpt2",
    seeds: List[int] = [0],
    prompts: int = 50,
    max_len: int = 128,
    config: Optional[ProbCalConfig] = None,
) -> Dict:
    cfg = config or ProbCalConfig()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()

    seed_reports = []
    for sd in seeds:
        seed_everything(sd)
        texts = _make_prompts(prompts, sd)
        split = max(1, int(len(texts) * cfg.calib_frac))
        cal_texts = texts[:split]
        test_texts = texts[split:]
        ood_texts = _make_ood(test_texts)

        t0 = time.perf_counter()
        nll_base, H_base = _nll_and_entropy(model, tok, test_texts, max_len, cfg.topk_eval, tau=None)
        nll_base_ood, _ = _nll_and_entropy(model, tok, ood_texts, max_len, cfg.topk_eval, tau=None)
        tau = _fit_temperature(model, tok, cal_texts, max_len, cfg.steps, cfg.lr)
        nll_post, H_post = _nll_and_entropy(model, tok, test_texts, max_len, cfg.topk_eval, tau=tau, alpha_curv=cfg.alpha_curv)
        nll_post_ood, _ = _nll_and_entropy(model, tok, ood_texts, max_len, cfg.topk_eval, tau=tau, alpha_curv=cfg.alpha_curv)
        dt = time.perf_counter() - t0

        seed_reports.append(
            {
                "seed": sd,
                "tau": tau,
                "alpha_curv": cfg.alpha_curv,
                "nll_base": nll_base,
                "nll_post": nll_post,
                "nll_base_ood": nll_base_ood,
                "nll_post_ood": nll_post_ood,
                "entropy_base": H_base,
                "entropy_post": H_post,
                "eval_time_s": dt,
            }
        )

    def mean(k: str) -> float:
        vals = [r[k] for r in seed_reports]
        return float(sum(vals) / max(1, len(vals)))

    summary = {
        "mean_tau": mean("tau"),
        "mean_nll_base": mean("nll_base"),
        "mean_nll_post": mean("nll_post"),
        "mean_nll_base_ood": mean("nll_base_ood"),
        "mean_nll_post_ood": mean("nll_post_ood"),
        "mean_entropy_base": mean("entropy_base"),
        "mean_entropy_post": mean("entropy_post"),
    }

    return {
        "model": model_name,
        "seeds": seeds,
        "prompts": prompts,
        "max_len": max_len,
        "config": asdict(cfg),
        "summary": summary,
        "seeds_report": seed_reports,
    }


def run_probcal_cli():
    import argparse

    p = argparse.ArgumentParser(description="REVO Phase VII: Deep probabilistic calibration (temperature + local curvature)")
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--prompts", type=int, default=50)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--out", type=str, default="quality/final_runs/probcal_report.json")

    p.add_argument("--calib_frac", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--alpha_curv", type=float, default=0.1)
    p.add_argument("--topk_eval", type=int, default=64)

    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = ProbCalConfig(
        calib_frac=args.calib_frac,
        steps=args.steps,
        lr=args.lr,
        alpha_curv=args.alpha_curv,
        topk_eval=args.topk_eval,
    )

    report = evaluate_probcal(
        model_name=args.model,
        seeds=seeds,
        prompts=args.prompts,
        max_len=args.max_len,
        config=cfg,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"results_path": args.out}))


if __name__ == "__main__":
    run_probcal_cli()
