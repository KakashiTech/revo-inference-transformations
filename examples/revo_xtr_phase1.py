from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Any

import numpy as np
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from revo._utils import count_parameters, evaluate_nll, free_memory_trim, measure_memory_rss
from revo.layer_profile import profile_model_2d, allocate_ranks_energy_with_caps
from revo.lowrank import replace_2d_modules_with_lowrank
from revo.archive.holomorphic_projection import holomorphic_project_model, adjust_ranks_geodesic


def _texts_default() -> List[str]:
    return [
        "Hola REVO! Resume en dos líneas qué es una descomposición de rango bajo.",
        "Explain in one sentence what spectral truncation does to a weight matrix.",
        "Lista tres ventajas de aproximar capas densas con TT/MPO.",
        "¿Por qué calibrar (healing) tras truncamiento SVD puede estabilizar precisión?",
        "Give a short definition of effective rank and its relation to information energy.",
    ]





def _ein_curvature_stats(profile: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    vals = []
    for _, p in profile.items():
        s = p.get("singular_values")
        if isinstance(s, np.ndarray) and s.size > 1:
            d = np.diff(s.astype(np.float64))
            c = float(np.mean(np.abs(d)))
            vals.append(c)
    return {
        "ein_curv_mean": float(np.mean(vals) if vals else 0.0),
        "ein_curv_std": float(np.std(vals) if vals else 0.0),
    }


def _collect_module_inputs(
    model: torch.nn.Module,
    tok,
    texts: List[str],
    module_names: List[str],
    max_length: int = 128,
    max_per_module: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Collect real input activations for specific modules via forward_pre hooks.
    Returns mapping name -> Tensor[N, in_features].
    """
    want = set(module_names)
    bufs: Dict[str, List[torch.Tensor]] = {n: [] for n in want}
    hooks = []

    def _pre_hook(name: str):
        def fn(mod, inputs):
            try:
                x = inputs[0]
                if isinstance(x, torch.Tensor):
                    x_flat = x.detach().to(device="cpu", dtype=torch.float32)
                    if x_flat.dim() > 2:
                        x_flat = x_flat.view(-1, x_flat.shape[-1])
                    bufs[name].append(x_flat)
            except Exception:
                pass
        return fn

    # Register hooks on desired modules
    name_to_module: Dict[str, torch.nn.Module] = {n: m for n, m in model.named_modules() if n in want}
    for n, m in name_to_module.items():
        try:
            hooks.append(m.register_forward_pre_hook(lambda mod, inp, _n=n: _pre_hook(_n)(mod, inp)))
        except Exception:
            continue

    # Run a few prompts through the model to trigger hooks
    model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
            _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)

    # Remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    out: Dict[str, torch.Tensor] = {}
    for n, parts in bufs.items():
        if not parts:
            continue
        X = torch.cat(parts, dim=0)
        if X.shape[0] > max_per_module:
            X = X[: max_per_module]
        out[n] = X
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="REVO XTR Phase 1: EinFields + Geodesic + Holomorphic + Mesh-Elastic")
    ap.add_argument("--model", default=os.environ.get("REVO_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("REVO_SEED", "0") or 0))
    ap.add_argument("--energy-keep", type=float, default=0.98)
    ap.add_argument("--max-rank", type=int, default=None)
    ap.add_argument("--max-rank-frac", type=float, default=0.25)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--results-json", default=None)
    ap.add_argument("--xtr-name", default="REVO XTR")
    ap.add_argument("--xtr-holo", action="store_true")
    ap.add_argument("--xtr-include-patterns", nargs="*", default=["c_fc", "c_proj"], help="Only apply XTR ops (holo/rank replace) to modules whose names include any of these patterns. Default targets MLP (c_fc,c_proj)")
    ap.add_argument("--xtr-geodesic-strength", type=float, default=0.35)
    ap.add_argument("--xtr-mesh-elastic", action="store_true")
    ap.add_argument("--xtr-target-rss-mb", type=float, default=1024.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    rss0 = measure_memory_rss()
    t0 = time.perf_counter()
    base_nll = evaluate_nll(model, tok, _texts_default(), max_length=args.max_length)
    base_time = time.perf_counter() - t0
    base_params = count_parameters(model)
    rss_base = measure_memory_rss()

    prof = profile_model_2d(model, name_patterns=list(args.xtr_include_patterns) if args.xtr_include_patterns else None, max_rank=args.max_rank)
    ein_stats = _ein_curvature_stats(prof)

    if args.xtr_holo:
        holo_layers = holomorphic_project_model(model, name_patterns=list(args.xtr_include_patterns) if args.xtr_include_patterns else None)
    else:
        holo_layers = 0

    ranks = allocate_ranks_energy_with_caps(
        prof,
        energy_keep=args.energy_keep,
        max_rank=args.max_rank,
        max_rank_frac=args.max_rank_frac,
    )
    ranks_adj = adjust_ranks_geodesic(prof, ranks, strength=float(args.xtr_geodesic_strength))

    # Filter ranks to included patterns only
    if args.xtr_include_patterns:
        pats = set(args.xtr_include_patterns)
        ranks_adj = {k: v for k, v in ranks_adj.items() if any(p in k for p in pats)}

    # Establish max_length for calibration/eval; mesh-elastic may reduce later for eval
    target_mb = float(args.xtr_target_rss_mb)
    ml = int(args.max_length)

    # Collect real inputs for modules to be replaced to improve calibration fidelity
    module_inputs = _collect_module_inputs(
        model,
        tok,
        _texts_default(),
        list(ranks_adj.keys()),
        max_length=ml,
        max_per_module=max(1024, int(args.calib_samples)),
    ) if ranks_adj else {}

    rep = replace_2d_modules_with_lowrank(
        model,
        ranks_adj,
        calibrate=bool(args.calibrate),
        calibrate_samples=int(args.calib_samples),
        seed=int(args.seed),
        module_inputs=module_inputs,
    )

    # Free calibration inputs and collect garbage before measuring RSS
    module_inputs = None
    try:
        gc.collect()
    except Exception:
        pass
    free_memory_trim()

    if args.xtr_mesh_elastic:
        rss1 = measure_memory_rss()
        if rss1 > 0:
            cur_mb = float(rss1) / (1024.0 * 1024.0)
            if cur_mb > target_mb:
                factor = max(0.25, min(1.0, target_mb / max(1e-6, cur_mb)))
                ml = max(32, int(ml * factor))

    t1 = time.perf_counter()
    comp_nll = evaluate_nll(model, tok, _texts_default(), max_length=ml)
    comp_time = time.perf_counter() - t1
    comp_params = count_parameters(model)
    # Trim after eval to release transient buffers before measuring
    try:
        gc.collect()
    except Exception:
        pass
    free_memory_trim()
    rss_comp = measure_memory_rss()

    out = {
        "variant": str(args.xtr_name),
        "phase": "Phase1-XTR",
        "model": str(args.model),
        "seed": int(args.seed),
        "texts_n": len(_texts_default()),
        "xtr": {
            "holomorphic": bool(args.xtr_holo),
            "holo_layers_modified": int(holo_layers),
            "geodesic_strength": float(args.xtr_geodesic_strength),
            "mesh_elastic": bool(args.xtr_mesh_elastic),
            "target_rss_mb": float(args.xtr_target_rss_mb),
        },
        "ein_curvature": ein_stats,
        "baseline": {
            "nll": float(base_nll),
            "eval_time_s": float(base_time),
            "params": int(base_params),
            "rss_bytes": int(rss_base),
        },
        "compressed": {
            "nll": float(comp_nll),
            "eval_time_s": float(comp_time),
            "params": int(comp_params),
            "rss_bytes": int(rss_comp),
        },
        "alloc": {
            "initial": {k: int(v) for k, v in ranks.items()},
            "geodesic": {k: int(v) for k, v in ranks_adj.items()},
            "replaced": {k: [int(a), int(b)] for k, (a, b) in rep.items()},
        },
        "timestamp": int(time.time()),
    }

    out_path = args.results_json
    if out_path is None:
        ts = int(time.time())
        out_dir = os.path.join("quality", "phase1_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"revo_xtr_phase1_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    summary = {
        "variant": str(args.xtr_name),
        "nll_delta": float(out["compressed"]["nll"] - out["baseline"]["nll"]),
        "eval_time_ratio": float(out["compressed"]["eval_time_s"] / max(1e-9, out["baseline"]["eval_time_s"])),
        "params_ratio": float(out["compressed"]["params"] / max(1, out["baseline"]["params"])),
        "rss_delta_bytes": int(out["compressed"]["rss_bytes"] - out["baseline"]["rss_bytes"]),
        "ein_curv_mean": float(out["ein_curvature"]["ein_curv_mean"]),
        "results_path": out_path,
    }
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
