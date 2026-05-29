import json
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
        "Context evokes only the necessary structures.",
        "Only local perturbations: no global forward is required.",
        "Latent memory is accessed, not stored explicitly.",
        "Activation by resonance, not by schedule.",
        "The model is a field, not an object.",
        "Explain how locality can approximate full inference.",
        "Relate sparse activation to cognitive economy.",
        "When is activation unnecessary?",
        "What signals increase resonance?",
        "How to design implicit existence in practice?",
    ]
    prompts = []
    for i in range(n):
        s = base[i % len(base)]
        if rng.random() < 0.5:
            s += f" [seed={seed} v={rng.randint(0,9999)}]"
        prompts.append(s)
    return prompts


@dataclass
class ImplicitConfig:
    codebook_k: int = 16
    iters: int = 5
    q_quantile: float = 0.5  # Fraction below this quantile deemed "activated"


@torch.no_grad()
def _token_embeddings(model, tok, texts: List[str], max_length: int) -> torch.Tensor:
    embs = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(DEVICE)
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple of [1, T, H]
        last = hs[-1][0]  # [T, H]
        embs.append(last)
    return torch.cat(embs, dim=0)  # [N_tokens, H]


@torch.no_grad()
def _kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    # Simple k-means on CPU
    rng = torch.Generator(device=x.device)
    rng.manual_seed(seed)
    k_actual = min(k, x.size(0))
    if k_actual < k:
        import warnings
        warnings.warn(f"Codebook k={k} > available tokens {x.size(0)}, using k={k_actual}")
    idx = torch.randperm(x.size(0), generator=rng)[:k_actual]
    c = x[idx].clone()  # [k_actual, H]
    for _ in range(max(1, iters)):
        # Assign
        d = torch.cdist(x, c)  # [N, k]
        a = torch.argmin(d, dim=1)  # [N]
        # Update
        for j in range(k_actual):
            sel = x[a == j]
            if sel.numel() > 0:
                c[j] = sel.mean(dim=0)
    return c


@torch.no_grad()
def _nearest_distances(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(x, c)  # [N, k]
    m = torch.min(d, dim=1).values
    return m  # [N]


@torch.no_grad()
def _eval_nll(model, tok, texts: List[str], max_length: int) -> float:
    total = 0.0
    tokens = 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(DEVICE)
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = float(out.loss.item())
        tokens += int(input_ids.numel())
        total += loss * int(input_ids.numel())
    return float(total / max(1, tokens))


def evaluate_implicit(
    model_name: str = "sshleifer/tiny-gpt2",
    seeds: List[int] = [0],
    prompts: int = 50,
    max_len: int = 128,
    config: Optional[ImplicitConfig] = None,
) -> Dict:
    cfg = config or ImplicitConfig()
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
        t0 = time.perf_counter()
        x = _token_embeddings(model, tok, texts, max_len)
        # Limit to avoid excessive memory on large texts
        if x.size(0) > 20000:
            x = x[:20000]
        c = _kmeans(x, k=max(2, min(cfg.codebook_k, x.size(0))), iters=max(1, cfg.iters), seed=sd)
        d = _nearest_distances(x, c)
        thr = torch.quantile(d, q=min(0.99, max(0.01, cfg.q_quantile)))
        activated = (d <= thr).float().mean().item()
        nll = _eval_nll(model, tok, texts, max_len)
        dt = time.perf_counter() - t0
        seed_reports.append(
            {
                "seed": sd,
                "tokens": int(x.size(0)),
                "codebook_k": int(c.size(0)),
                "quantile": float(cfg.q_quantile),
                "distance_thr": float(thr.item()),
                "activated_fraction": float(activated),
                "nll": nll,
                "eval_time_s": dt,
            }
        )

    def mean(k: str) -> float:
        vals = [r[k] for r in seed_reports]
        return float(sum(vals) / max(1, len(vals)))

    summary = {
        "mean_activated_fraction": mean("activated_fraction"),
        "mean_nll": mean("nll"),
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


def run_implicit_cli():
    import argparse

    p = argparse.ArgumentParser(description="REVO Phase VIII: Implicit model existence via resonance-driven sparse activation plan")
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--prompts", type=int, default=50)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--out", type=str, default="quality/final_runs/implicit_report.json")

    p.add_argument("--codebook_k", type=int, default=16)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--q_quantile", type=float, default=0.5)

    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = ImplicitConfig(codebook_k=args.codebook_k, iters=args.iters, q_quantile=args.q_quantile)

    report = evaluate_implicit(
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
    run_implicit_cli()
