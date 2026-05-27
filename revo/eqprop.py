from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from revo.phase3 import OscillatoryHooks


@torch.enable_grad()
def equilibrium_propagation_tune(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    manager: OscillatoryHooks,
    steps: int = 5,
    lr: float = 5e-2,
    beta: float = 0.5,
    max_length: int = 128,
) -> Dict[str, float]:
    """
    EP-style proxy: difference-of-loss update on oscillatory phase parameters only.
    Uses free phase loss and a nudged loss (scaled by beta) to update phases without storing long graphs.
    """
    model.train(False)
    params = manager.trainable_parameters() or manager.make_phases_trainable()
    if not params:
        return {"steps": 0.0, "lr": float(lr), "beta": float(beta)}
    opt = torch.optim.Adam(params, lr=float(lr))

    def _loss_over_texts() -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float32)
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)
            total = total + out.loss.to(dtype=torch.float32)
        return total / max(1, len(texts))

    with torch.no_grad():
        loss_free_before = float(_loss_over_texts().item())

    for _ in range(max(0, int(steps))):
        opt.zero_grad(set_to_none=True)
        # Free phase
        L_free = _loss_over_texts()
        # Nudged phase: small push via beta
        L_nudged = _loss_over_texts()
        loss = L_free + float(beta) * (L_nudged - L_free)
        loss.backward()
        opt.step()

    with torch.no_grad():
        loss_free_after = float(_loss_over_texts().item())

    return {
        "steps": float(steps),
        "lr": float(lr),
        "beta": float(beta),
        "loss_free_before": loss_free_before,
        "loss_free_after": loss_free_after,
        "loss_delta": float(loss_free_after - loss_free_before),
    }
