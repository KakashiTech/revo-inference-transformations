import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import seed_everything, DEVICE


def _make_prompts(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    base = [
        "Explain the principle of least action in simple terms.",
        "Translate to Spanish: The patterns within patterns form meaning.",
        "Summarize the plot of a classic novel in one paragraph.",
        "Write a short haiku about winter and memory.",
        "List three applications of Fourier transforms in engineering.",
        "Given two vectors u and v, explain the geometric meaning of the dot product.",
        "What are the tradeoffs of strong encryption in distributed systems?",
        "Draft a polite email to reschedule a meeting due to illness.",
        "Provide a step-by-step derivation of Bayes' theorem.",
        "Describe the differences between compile-time and runtime polymorphism.",
    ]
    prompts = []
    for i in range(n):
        # Mix curated and lightly perturbed prompts
        s = base[i % len(base)]
        if rng.random() < 0.5:
            s += f" Seed:{seed} Variant:{rng.randint(0, 9999)}"
        prompts.append(s)
    return prompts


@dataclass
class RegimeConfig:
    # Thresholds control passive vs active regime.
    # We avoid hard switches; these shape a smooth logistic.
    density_weight: float = 0.5
    depth_weight: float = 0.5
    size_weight: float = 0.25
    center: float = 0.6
    sharpness: float = 4.0
    topk_entropy: int = 32
    layer_name_filters: Tuple[str, ...] = (
        "attn.c_attn",
        "mlp.c_fc",
        "mlp.c_proj",
    )


@torch.no_grad()
def _n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _semantic_density_and_depth(
    model: AutoModelForCausalLM,
    tokenizer,
    texts: List[str],
    max_length: int,
    topk: int,
) -> Tuple[float, float]:
    model.eval()
    device = _device()

    densities: List[float] = []
    depths: List[float] = []

    for t in texts:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = out.logits  # [1, T, V]
        hs = out.hidden_states  # tuple(len=L+1) of [1, T, H]

        # Semantic density proxy: margin between top1 and top2 averaged over tokens
        # (higher margin => higher density of meaning / confidence)
        topk_vals, _ = torch.topk(logits[0], k=min(topk, logits.size(-1)), dim=-1)
        margins = (topk_vals[:, 0] - topk_vals[:, 1]).clamp(min=0)
        density = torch.tanh(margins.mean() / 2.0).item()
        densities.append(float(density))

        # Relational depth proxy: average representational change across layers
        # Normalize by hidden size to avoid scale effects.
        if len(hs) >= 2:
            diffs = []
            for li in range(1, len(hs)):
                prev = hs[li - 1][0]  # [T, H]
                cur = hs[li][0]
                # Mean cosine distance across tokens
                a = F.normalize(prev, dim=-1)
                b = F.normalize(cur, dim=-1)
                cos = (a * b).sum(dim=-1).mean()
                diffs.append(float((1.0 - cos).clamp(min=0.0, max=2.0).item()))
            depth = float(sum(diffs) / max(1, len(diffs)))
        else:
            depth = 0.0
        depths.append(depth)

    return float(sum(densities) / max(1, len(densities))), float(
        sum(depths) / max(1, len(depths))
    )


def _smooth_activation(x: float, center: float, sharpness: float) -> float:
    # Smooth logistic from 0 to 1 centered at 'center'
    z = sharpness * (x - center)
    return 1.0 / (1.0 + math.exp(-z))


def _build_gamma_map(
    model: nn.Module,
    strength: float,
    name_filters: Tuple[str, ...],
) -> Dict[str, float]:
    # Recommend gamma per layer name for ResonanceGateWrap-compatible layers
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    selected = [n for n in names if any(k in n for k in name_filters)]
    gamma_map: Dict[str, float] = {}
    # Depth-aware ramp: deeper layers receive slightly stronger activation
    selected_sorted = sorted(selected, key=lambda s: (len(s), s))
    L = max(1, len(selected_sorted))
    for i, name in enumerate(selected_sorted):
        depth_gain = 0.5 + 0.5 * (i / (L - 1) if L > 1 else 0.0)
        gamma = float(max(0.0, min(1.0, strength * depth_gain)))
        gamma_map[name] = gamma
    return gamma_map


def evaluate_regimes(
    model_name: str = "sshleifer/tiny-gpt2",
    seeds: List[int] = [0],
    prompts: int = 50,
    max_len: int = 128,
    config: Optional[RegimeConfig] = None,
) -> Dict:
    cfg = config or RegimeConfig()
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    params = _n_params(model)
    size_term = math.log10(max(1.0, float(params))) / 10.0  # ~0..1 for small–mid models

    seed_reports = []
    for sd in seeds:
        seed_everything(sd)
        texts = _make_prompts(prompts, sd)
        t0 = time.perf_counter()
        density, depth = _semantic_density_and_depth(
            model, tokenizer, texts, max_len, cfg.topk_entropy
        )
        # Composite regime index
        x = (
            cfg.density_weight * density
            + cfg.depth_weight * depth
            + cfg.size_weight * size_term
        ) / (cfg.density_weight + cfg.depth_weight + cfg.size_weight)
        strength = _smooth_activation(x, cfg.center, cfg.sharpness)
        gamma_map = _build_gamma_map(model, strength, cfg.layer_name_filters)
        dt = time.perf_counter() - t0
        seed_reports.append(
            {
                "seed": sd,
                "density": density,
                "depth": depth,
                "size_term": size_term,
                "regime_index": x,
                "gating_strength": strength,
                "gamma_map_count": len(gamma_map),
                "eval_time_s": dt,
            }
        )

    mean_strength = float(sum(r["gating_strength"] for r in seed_reports) / len(seed_reports))
    regime = "micro-passive" if mean_strength < 0.33 else ("macro-active" if mean_strength > 0.66 else "transition-smooth")

    return {
        "model": model_name,
        "params": int(params),
        "seeds": seeds,
        "prompts": prompts,
        "max_len": max_len,
        "config": asdict(cfg),
        "summary": {
            "mean_gating_strength": mean_strength,
            "regime": regime,
        },
        "seeds_report": seed_reports,
    }


def run_regimes_cli():
    import argparse

    parser = argparse.ArgumentParser(description="REVO Phase VI: Regimes of existence (micro↔macro) with smooth activation")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--prompts", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--out", type=str, default="quality/final_runs/regimes_report.json")

    parser.add_argument("--density_weight", type=float, default=0.5)
    parser.add_argument("--depth_weight", type=float, default=0.5)
    parser.add_argument("--size_weight", type=float, default=0.25)
    parser.add_argument("--center", type=float, default=0.6)
    parser.add_argument("--sharpness", type=float, default=4.0)
    parser.add_argument("--topk_entropy", type=int, default=32)

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cfg = RegimeConfig(
        density_weight=args.density_weight,
        depth_weight=args.depth_weight,
        size_weight=args.size_weight,
        center=args.center,
        sharpness=args.sharpness,
        topk_entropy=args.topk_entropy,
    )

    report = evaluate_regimes(
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
    run_regimes_cli()
