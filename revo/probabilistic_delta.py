from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _last_hidden_states(model: nn.Module, tokenizer, texts: List[str], max_length: int = 128) -> torch.Tensor:
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
            last_h = out.hidden_states[-1][:, -1, :].detach().to(dtype=torch.float32)
            outs.append(last_h)
    return torch.cat(outs, dim=0)  # [N, D]


def _to_prob_and_scale(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    s = float(x.abs().max().item()) if x.numel() > 0 else 1.0
    s = max(s, 1e-8)
    p = (x / s + 1.0) * 0.5
    p = p.clamp(0.0, 1.0)
    return p, s


@torch.no_grad()
def pdm_eval_lm_head(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    bits: int = 32,
    topk: int = 256,
    max_length: int = 128,
) -> Dict[str, float]:
    """Approximate lm_head matmul using stochastic bitstreams (PDM proxy) on top-K vocab rows.

    Returns cosine similarity between exact and PDM-approximated logits (restricted to top-K rows),
    and the wall-clock time for the PDM computation.
    """
    # Collect inputs and weights
    hs = _last_hidden_states(model, tokenizer, texts, max_length=max_length)  # [N, D]
    lm = getattr(model, "lm_head", None)
    if not isinstance(lm, nn.Module) or not hasattr(lm, "weight"):
        return {"pdm_cos": 0.0, "pdm_time_s": 0.0, "pdm_topk": 0.0, "bits": float(bits)}
    W = lm.weight.detach().to(dtype=torch.float32)  # [V, D]
    V, D = int(W.shape[0]), int(W.shape[1])
    K = min(int(topk), V)
    idx = torch.arange(K, dtype=torch.long)
    Wk = W.index_select(dim=0, index=idx).contiguous()  # [K, D]

    # Exact logits restricted to K rows
    exact = hs @ Wk.t()  # [N, K]

    # PDM approximation setup
    pW, sW = _to_prob_and_scale(Wk)  # [K, D]

    t0 = time.perf_counter()
    pdm_outs: List[torch.Tensor] = []
    B = int(max(1, bits))
    for n in range(hs.shape[0]):
        x = hs[n]  # [D]
        px, sx = _to_prob_and_scale(x)
        # Sample bitstreams
        # Xpm: [B, D], values in {-1, +1}
        Xpm = (torch.rand((B, D)) < px.unsqueeze(0)).to(dtype=torch.float32) * 2.0 - 1.0
        # Wpm: [B, K, D]
        Wpm = (torch.rand((B, K, D)) < pW.unsqueeze(0)).to(dtype=torch.float32) * 2.0 - 1.0
        # Multiply and sum over feature dimension -> [B, K]
        prod = (Xpm.unsqueeze(1) * Wpm).sum(dim=2)
        y_hat = float(sx * sW) * prod.mean(dim=0)  # [K]
        pdm_outs.append(y_hat.unsqueeze(0))
    pdm = torch.cat(pdm_outs, dim=0)  # [N, K]
    pdm_time = time.perf_counter() - t0

    # Cosine similarity between rows
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a / (a.norm(dim=1, keepdim=True) + 1e-9)
        b = b / (b.norm(dim=1, keepdim=True) + 1e-9)
        return float((a * b).sum(dim=1).mean().item())

    cos = _cos(exact, pdm)
    return {"pdm_cos": cos, "pdm_time_s": float(pdm_time), "pdm_topk": float(K), "bits": float(B)}
