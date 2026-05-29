from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import seed_everything, DEVICE


@dataclass
class CategoryConfig:
    max_len: int = 128
    prompts: int = 100


class Context:
    def __init__(self, text: str):
        self.text = text
        self.meta: Dict[str, Any] = {}


class Morphism:
    def __init__(self, name: str, fn: Callable[[Context], None]):
        self.name = name
        self.fn = fn

    def __call__(self, ctx: Context) -> None:
        self.fn(ctx)


class Compose(Morphism):
    def __init__(self, name: str, morphisms: List[Morphism]):
        super().__init__(name, lambda ctx: None)
        self.morphisms = morphisms

    def __call__(self, ctx: Context) -> None:
        for m in self.morphisms:
            m(ctx)


class Parallel(Morphism):
    def __init__(self, name: str, morphisms: List[Morphism]):
        super().__init__(name, lambda ctx: None)
        self.morphisms = morphisms

    def __call__(self, ctx: Context) -> None:
        # Run all, merge results conservatively (shortest) to preserve NLL semantics
        texts = []
        metas = []
        for m in self.morphisms:
            c = Context(ctx.text)
            c.meta = dict(ctx.meta)
            m(c)
            texts.append(c.text)
            metas.append(c.meta)
        # choose the variant with minimal edit distance to original to avoid perturbing semantics
        best_i = int(np.argmin([_levenshtein(ctx.text, t) for t in texts]))
        ctx.text = texts[best_i]
        ctx.meta.update(metas[best_i])


class Conditional(Morphism):
    def __init__(self, name: str, predicate: Callable[[Context], bool], then_m: Morphism, else_m: Morphism | None = None):
        super().__init__(name, lambda ctx: None)
        self.predicate = predicate
        self.then_m = then_m
        self.else_m = else_m

    def __call__(self, ctx: Context) -> None:
        if self.predicate(ctx):
            self.then_m(ctx)
        elif self.else_m is not None:
            self.else_m(ctx)


def _levenshtein(a: str, b: str) -> int:
    # Simple DP edit distance to gate aggressive rewrites
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


# Primitive morphisms operating on text in a minimally invasive way

def morph_normalize() -> Morphism:
    def fn(ctx: Context) -> None:
        ctx.text = " ".join(ctx.text.strip().split())
    return Morphism("normalize", fn)


def morph_outline() -> Morphism:
    def fn(ctx: Context) -> None:
        # Create a lightweight outline at the end to cue structure without changing semantics
        if "[Outline]" not in ctx.text and len(ctx.text) > 40:
            ctx.text = ctx.text + "\n[Outline]: 1) contexto 2) objetivo 3) pasos 4) salida"
            ctx.meta["outlined"] = True
    return Morphism("outline", fn)


def morph_tag_hierarchy() -> Morphism:
    def fn(ctx: Context) -> None:
        # Tag with hierarchy level markers, avoid modifying the core prompt
        if "[H1]" not in ctx.text:
            ctx.text = "[H1] " + ctx.text
            ctx.meta["hierarchical"] = True
    return Morphism("tag_hierarchy", fn)


def morph_compose_default() -> Morphism:
    return Compose("compose_default", [morph_normalize(), morph_tag_hierarchy(), morph_outline()])


# Evaluation utilities

def _load_model_tok(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(DEVICE)
    return model, tok


def _encode(tok, text: str, max_len: int) -> Dict[str, torch.Tensor]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    return {k: v.to(DEVICE) for k, v in enc.items()}


def _avg_nll(model, tok, texts: List[str], max_len: int) -> float:
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


def _texts_hierarchical(n: int) -> List[str]:
    base = [
        "Planifica y resuelve: calcular media y desviación de [2,4,4,6,8] y explicar el proceso.",
        "Resume luego compara: explica REVO vs Q+P y luego contrasta ventajas.",
        "Descompon y resuelve: derivar regla de la cadena y aplicar a f(g(x)).",
        "Analiza y sintetiza: extrae puntos clave y produce un esquema final.",
    ]
    return [base[i % len(base)] + f" [CAT H#{i}]" for i in range(n)]


def evaluate_category(
    model_name: str,
    seeds: List[int],
    prompts: int,
    max_len: int,
) -> Dict[str, object]:
    hier_texts = _texts_hierarchical(prompts)

    metrics = []
    model, tok = _load_model_tok(model_name)

    for seed in seeds:
        seed_everything(seed)
        plan = morph_compose_default()
        structured: List[str] = []
        for t in hier_texts:
            ctx = Context(t)
            plan(ctx)
            structured.append(ctx.text)
        nll_base = _avg_nll(model, tok, hier_texts, max_len)
        nll_struct = _avg_nll(model, tok, structured, max_len)
        conservative_frac = float(np.mean([_levenshtein(a, b) <= 32 for a, b in zip(hier_texts, structured)]))
        metrics.append({
            "name": f"category_seed{seed}",
            "nll_base": nll_base,
            "nll_structured": nll_struct,
            "delta_nll": float(nll_struct - nll_base),
            "conservative_frac": conservative_frac,
        })

    return {
        "model": model_name,
        "seeds": seeds,
        "prompts": prompts,
        "max_length": max_len,
        "variants": metrics,
    }


def run_category_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run Category Morphisms (Phase II.3)")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--prompts", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--results-json", default=None)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    result = evaluate_category(args.model, seeds, int(args.prompts), int(args.max_length))

    out_path = args.results_json or os.path.join("quality", "phase2_runs", f"ii3_category_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"results_path": out_path}, indent=2))


if __name__ == "__main__":
    run_category_cli()
