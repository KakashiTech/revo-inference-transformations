#!/usr/bin/env python3
"""Test Metric Field PINN — random embeddings, train, verify residual drops."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revo.metric_field_pinn import (
    MetricFieldPINN,
    curvature_metric,
    pde_residual,
    train_pinn,
)


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")

    N_TOKENS = 100
    DIM = 64
    ITERS = 3000
    LR = 1e-3
    HIDDEN = 256
    W0 = 10.0
    GRAD_PENALTY = 0.1
    FD_H = 0.1
    TARGET_RESIDUAL = 1e-3

    print(f"Generating {N_TOKENS} random {DIM}-dim embeddings ...")
    embeddings = torch.randn(N_TOKENS, DIM, device=device)

    model = MetricFieldPINN(in_features=DIM, hidden_dim=HIDDEN, w0=W0)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    model.eval()
    with torch.no_grad():
        loss_before = pde_residual(model, embeddings, h=FD_H).item()
    print(f"\nPDE residual BEFORE training: {loss_before:.6e}")

    print(f"\nTraining for {ITERS} iterations (lr={LR}, hidden={HIDDEN}, "
          f"w0={W0}, grad_penalty={GRAD_PENALTY}, scheduler=cosine) ...")
    model, history = train_pinn(
        model, embeddings,
        iters=ITERS, lr=LR,
        scheduler="cosine", min_lr=1e-6,
        weight_decay=1e-5,
        grad_penalty=GRAD_PENALTY,
        fd_h=FD_H,
        verbose=True,
    )

    model.eval()
    with torch.no_grad():
        loss_after = pde_residual(model, embeddings, h=FD_H).item()
    print(f"\nPDE residual AFTER  training: {loss_after:.6e}")
    ratio = loss_after / max(loss_before, 1e-30)
    print(f"Ratio after/before: {ratio:.6f}  (1.0 = no change)")

    if loss_after <= TARGET_RESIDUAL:
        print(f"✓ TARGET ACHIEVED: residual {loss_after:.6e} ≤ {TARGET_RESIDUAL}")
    else:
        print(f"✗ TARGET NOT ACHIEVED: residual {loss_after:.6e} > {TARGET_RESIDUAL}")
        print(f"  Best achievable residual for this config: {loss_after:.6e}")

    scores = curvature_metric(model, embeddings)
    print(f"\nCurvature scores (first 5): {scores[:5].tolist()}")

    out_dir = Path("quality/phase1_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "n_tokens": N_TOKENS,
        "dim": DIM,
        "iters": ITERS,
        "lr": LR,
        "hidden_dim": HIDDEN,
        "w0": W0,
        "grad_penalty": GRAD_PENALTY,
        "fd_h": FD_H,
        "scheduler": "cosine",
        "loss_before": loss_before,
        "loss_after": loss_after,
        "ratio_after_before": float(ratio),
        "target_achieved": loss_after <= TARGET_RESIDUAL,
        "target_residual": TARGET_RESIDUAL,
        "curvature_scores": scores.tolist(),
        "loss_history": history,
    }
    out_path = out_dir / "i1_metric_field_pinn.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\nFinal PDE residual: {loss_after:.6e}")
    print(f"Target ≤{TARGET_RESIDUAL}: {'YES' if loss_after <= TARGET_RESIDUAL else 'NO'}")


if __name__ == "__main__":
    main()
