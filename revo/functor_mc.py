from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _collect_last_hidden(model: nn.Module, tokenizer, texts: List[str], max_length: int = 128) -> torch.Tensor:
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
            last = out.hidden_states[-1][:, -1, :].detach().to(dtype=torch.float32)  # [B, D]
            outs.append(last)
    return torch.cat(outs, dim=0)  # [N, D]


def _functor_dot(x: torch.Tensor, w_row: torch.Tensor, bias: float) -> float:
    # Simulate MC primitive: scalar MAC accumulation in Python loop
    acc = 0.0
    xv = x.view(-1).tolist()
    wv = w_row.view(-1).tolist()
    for a, b in zip(xv, wv):
        acc += a * b
    acc += float(bias)
    return float(acc)


@torch.no_grad()
def verify_functor_mapping(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    name_patterns: List[str] | None = None,
    max_length: int = 128,
    max_rows: int = 64,
    max_samples: int = 16,
) -> Dict[str, float]:
    """Verify linear modules can be evaluated via functor (MC) primitives with numerical equivalence.

    Samples a subset of modules and output rows; compares functor result vs exact nn.Linear.
    Returns aggregate metrics.
    """
    X = _collect_last_hidden(model, tokenizer, texts, max_length=max_length)  # [N, D]
    if X.numel() == 0:
        return {"verified_modules": 0.0, "tested_rows_total": 0.0, "mean_abs_diff": 0.0, "max_abs_diff": 0.0}
    X = X[: max(1, int(max_samples)), :]

    pats = name_patterns or ["c_proj", "mlp", "attn"]
    tested_rows_total = 0
    diffs: List[float] = []
    verified_modules = 0

    for name, m in model.named_modules():
        if not any(p in name for p in pats):
            continue
        if not isinstance(m, nn.Linear):
            # try linear-like
            W = getattr(m, "weight", None)
            if not (isinstance(W, torch.Tensor) and W.dim() == 2):
                continue
        # Obtain linear-compatible handles
        W = getattr(m, "weight")
        b = getattr(m, "bias", None)
        if not (isinstance(W, torch.Tensor) and W.dim() == 2):
            continue
        out_dim, in_dim = int(W.shape[0]), int(W.shape[1])
        if in_dim != int(X.shape[1]):
            continue  # hidden size mismatch
        # Exact outputs
        try:
            Y = m(X) if isinstance(m, nn.Linear) else X @ W.t() + (b if isinstance(b, torch.Tensor) else 0.0)
        except Exception:
            continue
        rows = list(range(out_dim))
        random.shuffle(rows)
        rows = rows[: max(1, int(min(max_rows, out_dim)))]
        for r in rows:
            w_row = W[r].to(dtype=torch.float32)
            bias = float(b[r].item()) if isinstance(b, torch.Tensor) else 0.0
            for i in range(int(X.shape[0])):
                y_fun = _functor_dot(X[i], w_row, bias)
                y_ex = float(Y[i, r].item())
                diffs.append(abs(y_fun - y_ex))
                tested_rows_total += 1
        verified_modules += 1

    if not diffs:
        return {"verified_modules": float(verified_modules), "tested_rows_total": 0.0, "mean_abs_diff": 0.0, "max_abs_diff": 0.0}
    difft = torch.tensor(diffs, dtype=torch.float32)
    return {
        "verified_modules": float(verified_modules),
        "tested_rows_total": float(tested_rows_total),
        "mean_abs_diff": float(difft.mean().item()),
        "max_abs_diff": float(difft.max().item()),
    }
