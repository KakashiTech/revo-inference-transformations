from __future__ import annotations
from revo._logging import get_logger

from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn


def holomorphic_project_model(
    model: nn.Module,
    max_layers: int | None = None,
    name_patterns: list[str] | None = None,
) -> int:
    """
    Apply a holomorphic-inspired projection to 2D-weight modules (Linear or GPT-2 Conv1D-like):
    - Treat consecutive column pairs as complex components (A + i B)
    - Enforce equal norms and orthogonality within each pair (CR proxy)
    Returns number of layers modified.
    """
    n_modified = 0
    done = 0
    with torch.no_grad():
        for name, m in model.named_modules():
            if name_patterns is not None and not any(p in name for p in name_patterns):
                continue
            W = getattr(m, "weight", None)
            # require a 2D weight tensor; skip embeddings and non-tensor
            if not isinstance(W, torch.Tensor) or W.dim() != 2:
                continue
            if "embedding" in m.__class__.__name__.lower():
                continue
            # orient as [out, in]
            out_f, in_f = int(W.shape[0]), int(W.shape[1])
            if in_f < 2:
                continue
            # operate on column pairs
            pairs = in_f // 2
            if pairs == 0:
                continue
            # Build new W inplace
            for j in range(pairs):
                c0 = 2 * j
                c1 = 2 * j + 1
                a = W[:, c0].clone()
                b = W[:, c1].clone()
                # Orthonormalize (Gram-Schmidt)
                a_n = torch.norm(a).clamp_min(1e-12)
                a = a / a_n
                b = b - torch.dot(a, b) * a
                b_n = torch.norm(b).clamp_min(1e-12)
                b = b / b_n
                # Enforce equal norms (set to mean of original norms)
                target = float((a_n + b_n) * 0.5)
                # final columns
                W[:, c0] = a * target
                W[:, c1] = b * target
            n_modified += 1
            done += 1
            if max_layers is not None and done >= max_layers:
                break
    return n_modified


essential_profile_keys = {"entropy_norm", "effective_rank"}


def adjust_ranks_geodesic(profile: Dict[str, Dict[str, Any]],
                           base_ranks: Dict[str, int],
                           strength: float = 0.35) -> Dict[str, int]:
    """
    Geodesic-informed rank adjustment (proxy):
    - Use per-layer entropy_norm as a smoothness/curvature proxy.
    - Reduce rank on smoother layers (below mean entropy) proportionally to (mean - layer_entropy).
    """
    # collect entropy_norm across profiled layers
    ent_vals = []
    for name, p in profile.items():
        if "entropy_norm" in p:
            try:
                ent_vals.append(float(p.get("entropy_norm", 0.0)))
            except Exception:
                get_logger().warning("except Exception:")
    if not ent_vals:
        return dict(base_ranks)
    ent_vals = np.asarray(ent_vals, dtype=np.float64)
    mean_ent = float(ent_vals.mean())
    # adjust
    new_ranks: Dict[str, int] = {}
    for name, r in base_ranks.items():
        if r <= 1:
            new_ranks[name] = int(r)
            continue
        p = profile.get(name, {})
        le = float(p.get("entropy_norm", mean_ent))
        # gamma in [0, 1] for le <= mean, else 0
        gamma = max(0.0, (mean_ent - le) / max(mean_ent, 1e-12))
        scale = 1.0 - strength * gamma
        nr = max(1, int(round(r * scale)))
        new_ranks[name] = nr
    return new_ranks
