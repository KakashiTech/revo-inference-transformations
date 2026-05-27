import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import seed_everything, DEVICE
from revo.energy import measure_energy


def _make_prompts(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    base = [
        "Energy as a fundamental constraint enables sparse cognition.",
        "Persistence without activation: latent knowledge as field.",
        "Non-local information emerges via structured resonance.",
        "Capacity total exceeds the currently active state.",
        "Functional awareness: coherence across processing stages.",
        "Explain the trade-off between energy and inference depth.",
        "Why is latency a local phenomenon in a resonant system?",
        "Design a protocol for low-energy inference on CPU.",
        "Sketch how weak consciousness could be operationalized.",
        "Relate bio-computation to reversible logic constraints.",
    ]
    prompts = []
    for i in range(n):
        s = base[i % len(base)]
        if rng.random() < 0.5:
            s += f" [seed={seed} v={rng.randint(0,9999)}]"
        prompts.append(s)
    return prompts


@dataclass
class BioCompConfig:
    act_quantile: float = 0.75
    topk_eval: int = 64
    energy_warmup: int = 1
    energy_runs: int = 2


@torch.no_grad()
def _eval_nll(model, tok, texts: List[str], max_length: int) -> float:
    total = 0.0
    tokens = 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(_device())
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(_device())
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = float(out.loss.item())
        tokens += int(input_ids.numel())
        total += loss * int(input_ids.numel())
    return float(total / max(1, tokens))


@torch.no_grad()
def _hidden_metrics(model, tok, texts: List[str], max_length: int, act_q: float) -> Tuple[float, float]:
    # Returns (mean_active_fraction, mean_coherence)
    device = _device()
    active_fracs: List[float] = []
    coherences: List[float] = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple of [1, T, H]
        if len(hs) < 2:
            continue
        # Activation fraction across layers
        act_vals = []
        for h in hs:
            H = h[0]  # [T, H]
            magn = H.abs().reshape(-1)
            thr = torch.quantile(magn, q=min(0.99, max(0.01, act_q)))
            act = (magn >= thr).float().mean().item()
            act_vals.append(act)
        active_fracs.append(float(sum(act_vals) / max(1, len(act_vals))))
        # Coherence as mean cosine similarity between adjacent layers
        cos_vals = []
        for i in range(1, len(hs)):
            a = F.normalize(hs[i - 1][0], dim=-1)
            b = F.normalize(hs[i][0], dim=-1)
            cos = (a * b).sum(dim=-1).mean().item()
            cos_vals.append(float(cos))
        coherences.append(float(sum(cos_vals) / max(1, len(cos_vals))))
    return (
        float(sum(active_fracs) / max(1, len(active_fracs))) if active_fracs else 0.0,
        float(sum(coherences) / max(1, len(coherences))) if coherences else 0.0,
    )


def evaluate_biocomp(
    model_name: str = "sshleifer/tiny-gpt2",
    seeds: List[int] = [0],
    prompts: int = 50,
    max_len: int = 128,
    config: BioCompConfig | None = None,
) -> Dict:
    cfg = config or BioCompConfig()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(_device())
    model.eval()

    seed_reports = []
    for sd in seeds:
        seed_everything(sd)
        texts = _make_prompts(prompts, sd)
        t0 = time.perf_counter()
        nll = _eval_nll(model, tok, texts, max_len)
        act_frac, coherence = _hidden_metrics(model, tok, texts, max_len, cfg.act_quantile)
        # Energy proxy via Phase V module (CPU-only). Emulate warmup/runs.
        if cfg.energy_warmup > 0:
            _ = measure_energy(model, tok, texts[: min(len(texts), 5)], max_length=max_len)
        main_texts = texts * max(1, int(cfg.energy_runs))
        energy = measure_energy(model, tok, main_texts, max_length=max_len)
        dt = time.perf_counter() - t0
        seed_reports.append(
            {
                "seed": sd,
                "nll": nll,
                "active_fraction": float(act_frac),
                "coherence": float(coherence),
                "energy_report": {
                    "total_flops": energy.total_flops,
                    "dyn_energy_j": energy.dyn_energy_j,
                    "landauer_lower_j": energy.landauer_lower_j,
                    "latency_s": energy.latency_s,
                    "tokens": energy.tokens,
                },
                "eval_time_s": dt,
            }
        )

    def mean(k: str) -> float:
        vals = [r[k] for r in seed_reports]
        return float(sum(vals) / max(1, len(vals)))

    summary = {
        "mean_active_fraction": mean("active_fraction"),
        "mean_coherence": mean("coherence"),
        "mean_nll": mean("nll"),
        "mean_dyn_energy_j": float(sum(r["energy_report"]["dyn_energy_j"] for r in seed_reports) / max(1, len(seed_reports))),
        "mean_latency_s": float(sum(r["energy_report"]["latency_s"] for r in seed_reports) / max(1, len(seed_reports))),
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


def run_biocomp_cli():
    import argparse

    p = argparse.ArgumentParser(description="REVO Phase IX: Bio-computational convergence (energy, persistence, weak functional awareness)")
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--prompts", type=int, default=50)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--out", type=str, default="quality/final_runs/biocomp_report.json")

    p.add_argument("--act_quantile", type=float, default=0.75)
    p.add_argument("--topk_eval", type=int, default=64)
    p.add_argument("--energy_warmup", type=int, default=1)
    p.add_argument("--energy_runs", type=int, default=2)

    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = BioCompConfig(
        act_quantile=args.act_quantile,
        topk_eval=args.topk_eval,
        energy_warmup=args.energy_warmup,
        energy_runs=args.energy_runs,
    )

    report = evaluate_biocomp(
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
    run_biocomp_cli()
