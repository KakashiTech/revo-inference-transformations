from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class OscillatoryHooks:
    """Oscillatory gating manager for selected modules.

    Each selected module m gets a scalar phase phi_m. The forward output y is modulated as:
        y' = y * (1 + alpha * cos(phi_m + 2*pi*freq*t))
    where t is a global step (proxy for time/token index). For stability on CPU, we default to
    a static t=0 in gating; callers can increment t via step_tokens().

    Provides Kuramoto-style phase coupling updates for simple synchronization demos.
    """

    def __init__(self, alpha: float = 0.05, freq: float = 0.25) -> None:
        self.alpha = float(alpha)
        self.freq = float(freq)
        self._t: float = 0.0
        self._modules: Dict[str, nn.Module] = {}
        self._phases: Dict[str, torch.Tensor] = {}
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._phase_params: Dict[str, torch.nn.Parameter] = {}

    def attach(self, model: nn.Module, name_patterns: Optional[List[str]] = None, skip_lm_head: bool = True) -> List[str]:
        patterns = name_patterns or ["mlp", "c_fc", "c_proj", "attn"]
        selected: List[str] = []

        def _hook(name: str):
            def fn(mod, inp, out):
                try:
                    y = out
                    if not isinstance(y, torch.Tensor):
                        return out
                    phi = self._phases[name]
                    # Build differentiable gating using torch ops so phases can be trained
                    # Keep computation on y's device/dtype
                    phi_dev = phi.to(device=y.device, dtype=torch.float32)
                    t_tensor = torch.tensor(self._t, device=y.device, dtype=torch.float32)
                    phase_val = phi_dev + (2.0 * math.pi * float(self.freq)) * t_tensor
                    g = (1.0 + float(self.alpha) * torch.cos(phase_val)).to(device=y.device, dtype=y.dtype)
                    return y * g
                except Exception:
                    return out
            return fn

        for name, m in model.named_modules():
            if skip_lm_head and name == "lm_head":
                continue
            if not any(p in name for p in patterns):
                continue
            W = getattr(m, "weight", None)
            if not isinstance(W, torch.Tensor) or W.dim() != 2:
                continue
            try:
                self._modules[name] = m
                self._phases[name] = torch.tensor((torch.rand(()) * 2.0 - 1.0) * math.pi)
                h = m.register_forward_hook(_hook(name))
                self._hooks[name] = h
                selected.append(name)
            except Exception:
                continue
        return selected

    def detach(self) -> None:
        for h in list(self._hooks.values()):
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        self._modules.clear()
        self._phases.clear()

    def phases(self) -> Dict[str, float]:
        return {k: float(v.item()) for k, v in self._phases.items()}

    def set_phase_offset(self, delta: float) -> None:
        for k in self._phases:
            self._phases[k] = torch.tensor(float(self._phases[k].item() + delta))

    def add_phase_vector(self, deltas: Dict[str, float]) -> None:
        for k, dv in deltas.items():
            if k in self._phases:
                self._phases[k] = torch.tensor(float(self._phases[k].item() + dv))

    def reset_time(self) -> None:
        self._t = 0.0

    def step_tokens(self, n: int = 1) -> None:
        self._t += float(max(0, int(n)))

    def kuramoto_step(self, kappa: float = 0.0, dt: float = 0.1) -> None:
        """One Kuramoto update step with all-to-all uniform coupling.
        phi_i += dt * (omega_i + (kappa/N) * sum_j sin(phi_j - phi_i)). omega_i=0.
        """
        keys = list(self._phases.keys())
        if not keys or kappa == 0.0:
            return
        phis = torch.tensor([self._phases[k].item() for k in keys])
        N = float(phis.numel())
        # vectorized coupling term
        diff = phis.unsqueeze(0) - phis.unsqueeze(1)  # [N,N] phi_j - phi_i (will transpose sign below)
        # sum_j sin(phi_j - phi_i) = -sum_j sin(phi_i - phi_j)
        coupling = -torch.sin(diff).sum(dim=1)
        dphi = (float(kappa) / max(1.0, N)) * coupling
        phis = phis + float(dt) * dphi
        for i, k in enumerate(keys):
            self._phases[k] = torch.tensor(float(phis[i].item()))

    def kuramoto(self, steps: int = 0, kappa: float = 0.0, dt: float = 0.1) -> None:
        for _ in range(max(0, int(steps))):
            self.kuramoto_step(kappa=kappa, dt=dt)

    def phase_coherence(self) -> float:
        if not self._phases:
            return 0.0
        phis = torch.tensor([v.item() for v in self._phases.values()])
        z = torch.exp(1j * phis)
        R = torch.abs(z.mean()).item()
        return float(R)

    # --- Training support (Oscillatory BPTT proxy) ---
    def make_phases_trainable(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        self._phase_params.clear()
        for k, v in list(self._phases.items()):
            p = torch.nn.Parameter(v.detach().clone().to(dtype=torch.float32), requires_grad=True)
            self._phases[k] = p  # replace tensor with Parameter
            self._phase_params[k] = p
            params.append(p)
        return params

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return list(self._phase_params.values())


@torch.no_grad()
def interference_metrics(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    manager: OscillatoryHooks,
    delta_phi: float = 0.25,
    max_length: int = 128,
) -> Tuple[float, float]:
    """Compute cosine and symmetric KL between last-step logits under +delta and -delta phase offsets."""
    def _forward_last_logits() -> torch.Tensor:
        model.eval()
        outs: List[torch.Tensor] = []
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
            logits = out.logits  # [B, T, V]
            last = logits[:, -1, :].detach().to("cpu", dtype=torch.float32)
            outs.append(last)
            manager.step_tokens(int(enc.input_ids.shape[1]))
        return torch.cat(outs, dim=0)

    # Save/restore phases
    base_phases = manager.phases()
    # +delta
    manager.add_phase_vector({k: +float(delta_phi) for k in base_phases})
    Lp = _forward_last_logits()
    # -2*delta to go to -delta from +delta
    manager.add_phase_vector({k: -2.0 * float(delta_phi) for k in base_phases})
    Lm = _forward_last_logits()
    # Restore base
    manager.add_phase_vector({k: +float(delta_phi) for k in base_phases})

    # Cosine similarity
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.flatten(1)
        b = b.flatten(1)
        an = a / (a.norm(dim=1, keepdim=True) + 1e-9)
        bn = b / (b.norm(dim=1, keepdim=True) + 1e-9)
        return float((an * bn).sum(dim=1).mean().item())

    # Symmetric KL on softmax
    def _sym_kl(a: torch.Tensor, b: torch.Tensor) -> float:
        pa = torch.softmax(a, dim=-1) + 1e-9
        pb = torch.softmax(b, dim=-1) + 1e-9
        kl_ab = (pa * (pa.log() - pb.log())).sum(dim=-1).mean()
        kl_ba = (pb * (pb.log() - pa.log())).sum(dim=-1).mean()
        return float(0.5 * (kl_ab + kl_ba).item())

    return _cos(Lp, Lm), _sym_kl(Lp, Lm)


@torch.enable_grad()
def oscillatory_bptt_tune(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    manager: OscillatoryHooks,
    steps: int = 10,
    lr: float = 5e-2,
    kappa: float = 0.0,
    dt: float = 0.1,
    max_length: int = 128,
) -> Dict[str, float]:
    """Tune trainable phases via gradient descent on language modeling loss (proxy BPTT).

    Only phases (scalars per hooked module) are trained; model weights stay frozen.
    Optionally interleave Kuramoto coupling updates (kappa>0) between steps.
    """
    model.train(False)  # keep model in eval mode (no dropout), but allow grads through hooks
    params = manager.trainable_parameters() or manager.make_phases_trainable()
    if not params:
        return {"steps": 0.0, "lr": float(lr), "kappa": float(kappa)}
    opt = torch.optim.Adam(params, lr=float(lr))

    def _loss_over_texts() -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float32)
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=enc.input_ids)
            total = total + out.loss.to(dtype=torch.float32)
        return total / max(1, len(texts))

    with torch.no_grad():
        loss_before = float(_loss_over_texts().item())
        coh_before = float(manager.phase_coherence())
    for _ in range(max(0, int(steps))):
        opt.zero_grad(set_to_none=True)
        loss = _loss_over_texts()
        loss.backward()
        opt.step()
        if kappa != 0.0:
            manager.kuramoto_step(kappa=float(kappa), dt=float(dt))
    with torch.no_grad():
        loss_after = float(_loss_over_texts().item())
        coh_after = float(manager.phase_coherence())
    return {
        "steps": float(steps),
        "lr": float(lr),
        "kappa": float(kappa),
        "loss_before": loss_before,
        "loss_after": loss_after,
        "phase_coherence_before": coh_before,
        "phase_coherence_after": coh_after,
    }
