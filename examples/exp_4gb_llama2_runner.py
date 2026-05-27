from __future__ import annotations

import argparse
import json
import os
import random
import time
import resource
from typing import Dict, List, Any

import numpy as np


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _get_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = int(parts[1]) if len(parts) >= 2 else 0
                    return kb * 1024
    except Exception:
        pass
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(getattr(ru, "ru_maxrss", 0)) * 1024
    except Exception:
        return 0


def _prompts_general_10() -> List[str]:
    return [
        "Answer briefly: What is the capital of France?",
        "Explain in one sentence: why do seasons occur on Earth?",
        "Convert to JSON: name=Ana, age=19, city=Lima",
        "Translate to Spanish: The weather is sunny and mild today.",
        "List three prime numbers greater than 10.",
        "One-sentence summary: The fox jumps over the lazy dog.",
        "What is 17 times 13? Provide only the result.",
        "Give a short definition of entropy in information theory.",
        "Name two benefits of regular exercise.",
        "Paraphrase: Learning by doing helps retention.",
    ]


def _prompts_ood_small() -> List[str]:
    return [
        "def quicksort(a): return a if len(a)<2 else quicksort([x for x in a[1:] if x<=a[0]])+[a[0]]+quicksort([x for x in a[1:] if x>a[0]])",
        "<html><head><title>Test</title></head><body><p>Minimal page.</p></body></html>",
        "SELECT id, name FROM users WHERE active=1 ORDER BY created_at DESC LIMIT 5;",
    ]


def _llama_nll_tokens(llm, text: str, temperature: float, prompt_char_limit: int) -> tuple[float, int]:
    try:
        text = text[: int(prompt_char_limit)]
        llm.reset()
        out = llm.create_completion(
            prompt=text,
            max_tokens=1,
            echo=True,
            temperature=float(temperature),
            logprobs=1,
        )
        lp = out["choices"][0]["logprobs"]["token_logprobs"]
        usage = out.get("usage", {})
        n_prompt = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) and "prompt_tokens" in usage else max(0, len(lp) - 1)
        vals = [x for x in lp[:n_prompt] if x is not None]
        nll = float(-sum(vals) / max(1, len(vals)))
        return nll, int(n_prompt)
    except Exception as e:
        raise RuntimeError(f"llama-cpp scoring failed: {e}")


def _run_variant(
    llm,
    gen_texts: List[str],
    ood_texts: List[str],
    temperature: float,
    prompt_char_limit: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    peak_bytes = max(0, _get_rss_bytes())
    tot_tokens = 0
    nlls_gen: List[float] = []
    nlls_ood: List[float] = []
    cold = None
    warm_sum = 0.0
    warm_cnt = 0

    for i, t in enumerate(gen_texts):
        ts = time.perf_counter()
        nll_g, n_tok = _llama_nll_tokens(llm, t, temperature, prompt_char_limit)
        dt = time.perf_counter() - ts
        nlls_gen.append(nll_g)
        tot_tokens += int(n_tok)
        peak_bytes = max(peak_bytes, _get_rss_bytes())
        if cold is None:
            cold = dt
        else:
            warm_sum += dt
            warm_cnt += 1

    for t in ood_texts:
        nll_o, n_tok = _llama_nll_tokens(llm, t, temperature, prompt_char_limit)
        nlls_ood.append(nll_o)
        tot_tokens += int(n_tok)
        peak_bytes = max(peak_bytes, _get_rss_bytes())

    elapsed = time.perf_counter() - t0
    rss_mb = float(peak_bytes) / (1024.0 * 1024.0)
    tok_per_s = (float(tot_tokens) / elapsed) if elapsed > 0 else 0.0

    return {
        "nll_general": float(np.mean(nlls_gen)) if nlls_gen else None,
        "nll_ood": float(np.mean(nlls_ood)) if nlls_ood else None,
        "eval_time_s": float(elapsed),
        "tokens_scored": int(tot_tokens),
        "tokens_per_s": float(tok_per_s),
        "rss_peak_mb": float(rss_mb),
        "cold_start_s": float(cold or 0.0),
        "warm_avg_s": (float(warm_sum) / max(1, warm_cnt)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="Path to LLaMA-2-7B Q4_0 GGUF file")
    ap.add_argument("--outdir", default=os.path.join("quality", "compare"))
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.98)
    ap.add_argument("--prompt-char-limit", type=int, default=128)
    args = ap.parse_args()

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:
        raise RuntimeError(f"llama-cpp-python not available: {e}")

    os.makedirs(args.outdir, exist_ok=True)

    gen = _prompts_general_10()[: int(args.prompts)]
    ood = _prompts_ood_small()[: 3]

    _seed_everything(0)
    llm = Llama(
        model_path=args.gguf,
        n_ctx=int(args.ctx),
        n_threads=int(args.threads),
        n_batch=1,
        logits_all=True,
    )

    base = _run_variant(llm, gen, ood, temperature=1.0, prompt_char_limit=int(args.prompt_char_limit))
    revo = _run_variant(llm, gen, ood, temperature=float(args.tau), prompt_char_limit=int(args.prompt_char_limit))

    model_name = f"llama.cpp:{os.path.basename(args.gguf)}"

    base_payload = {
        "variant": "baseline",
        "model": model_name,
        "prompts": len(gen),
        **base,
    }
    revo_payload = {
        "variant": f"revo(tau={args.tau})",
        "model": model_name,
        "prompts": len(gen),
        **revo,
    }

    out_base = os.path.join(args.outdir, "exp_4gb_llama2_baseline.json")
    out_revo = os.path.join(args.outdir, "exp_4gb_llama2_revo_tau0.98.json")
    out_sum = os.path.join(args.outdir, "exp_4gb_llama2_summary.json")

    with open(out_base, "w", encoding="utf-8") as f:
        json.dump(base_payload, f, ensure_ascii=False, indent=2)
    with open(out_revo, "w", encoding="utf-8") as f:
        json.dump(revo_payload, f, ensure_ascii=False, indent=2)

    summary = {
        "model": model_name,
        "config": {
            "threads": int(args.threads),
            "ctx": int(args.ctx),
            "prompts": len(gen),
            "prompt_char_limit": int(args.prompt_char_limit),
        },
        "variants": [base_payload, revo_payload],
    }
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "baseline_path": out_base,
        "revo_path": out_revo,
        "summary_path": out_sum,
    }, indent=2))


if __name__ == "__main__":
    main()
