from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

KB = 1.380649e-23  # Boltzmann constant (J/K)
LN2 = math.log(2.0)
DEFAULT_T_K = 300.0
DEFAULT_E_PER_MAC_J = 3e-12  # heuristic picojoules per MAC

# -- RAPL / psutil globals --
_rapl_prev: Optional[float] = None
_cpu_ut_prev: Optional[float] = None
_cpu_wall_prev: Optional[float] = None


@dataclass
class LayerStats:
    name: str
    flops: float = 0.0
    bytes_read: float = 0.0
    bytes_written: float = 0.0


@dataclass
class EnergyReport:
    total_flops: float
    total_bytes_r: float
    total_bytes_w: float
    dyn_energy_j: float
    landauer_lower_j: float
    layers: Dict[str, LayerStats] = field(default_factory=dict)
    tokens: int = 0
    latency_s: float = 0.0


class _OpsHook:
    def __init__(self, name: str, stats: Dict[str, LayerStats]):
        self.name = name
        self.stats = stats
        self.last_in_shape: Tuple[int, ...] | None = None
        self.last_dtype: torch.dtype | None = None

    def pre(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return
        self.last_in_shape = tuple(x.shape)
        self.last_dtype = x.dtype

    def post(self, module: nn.Module, inputs: Tuple[Any, ...], output: torch.Tensor):
        if self.last_in_shape is None:
            return
        x_shape = self.last_in_shape
        y = output
        if not isinstance(y, torch.Tensor):
            return
        # Determine in/out features from weight orientation
        W = getattr(module, "weight")
        b = getattr(module, "bias", None)
        out_f, in_f = int(W.shape[0]), int(W.shape[1])
        if isinstance(b, torch.Tensor) and b.numel() == in_f:
            out_f, in_f = in_f, out_f  # transpose case
        # Determine tokens processed
        # Accept shapes like [B, T, in] or [N, in]
        if len(x_shape) >= 2:
            tokens = int(x_shape[-2]) if len(x_shape) >= 3 else int(x_shape[0])
        else:
            tokens = 1
        # FLOPs for matmul: 2 * in_f * out_f per token
        fl = float(2.0 * in_f * out_f * tokens)
        # Memory bytes (rough): read x, read W once per token, write y
        dtype_bytes = torch.tensor([], dtype=self.last_dtype).element_size() if self.last_dtype else 4
        br = float(tokens * (in_f) * dtype_bytes + (in_f * out_f) * dtype_bytes)
        bw = float(tokens * (out_f) * dtype_bytes)
        st = self.stats.setdefault(self.name, LayerStats(name=self.name))
        st.flops += fl
        st.bytes_read += br
        st.bytes_written += bw


@torch.no_grad()
def measure_energy(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
    T_K: float = DEFAULT_T_K,
    e_per_mac_j: float = DEFAULT_E_PER_MAC_J,
) -> EnergyReport:
    # Register hooks on linear-like modules
    stats: Dict[str, LayerStats] = {}
    hooks = []
    for name, m in model.named_modules():
        if hasattr(m, "weight") and isinstance(getattr(m, "weight"), torch.Tensor) and getattr(m, "weight").dim() == 2:
            h = _OpsHook(name, stats)
            hooks.append(m.register_forward_pre_hook(h.pre))
            hooks.append(m.register_forward_hook(h.post))

    model.eval()
    t0 = time.perf_counter()
    total_tokens = 0
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc.input_ids
            total_tokens += int(input_ids.numel())
            _ = model(input_ids=input_ids, attention_mask=enc.attention_mask, labels=input_ids)
    latency = time.perf_counter() - t0

    for h in hooks:
        h.remove()

    total_flops = sum(s.flops for s in stats.values())
    total_br = sum(s.bytes_read for s in stats.values())
    total_bw = sum(s.bytes_written for s in stats.values())
    dyn_energy = float(total_flops) * float(e_per_mac_j)
    landauer = float((total_bw * 8.0) * KB * T_K * LN2)
    return EnergyReport(
        total_flops=total_flops,
        total_bytes_r=total_br,
        total_bytes_w=total_bw,
        dyn_energy_j=dyn_energy,
        landauer_lower_j=landauer,
        layers=stats,
        tokens=total_tokens,
        latency_s=latency,
    )


# ---------------------------------------------------------------------------
# I.2 M-E-I Synchronisation: CPU power measurement & EnergyMonitor
# ---------------------------------------------------------------------------

def _read_rapl_joules() -> Optional[float]:
    """Sum RAPL package energy (J).  Returns None when unavailable."""
    total = 0.0
    found = False
    base = "/sys/class/powercap/intel-rapl"
    if not os.path.isdir(base):
        return None
    try:
        for entry in sorted(os.listdir(base)):
            d = os.path.join(base, entry)
            if not os.path.isdir(d):
                continue
            name_f = os.path.join(d, "name")
            uj_f = os.path.join(d, "energy_uj")
            if not os.path.isfile(name_f) or not os.path.isfile(uj_f):
                continue
            name = open(name_f).read().strip()
            if "package" in name:
                total += int(open(uj_f).read().strip())
                found = True
    except (PermissionError, FileNotFoundError, OSError, ValueError):
        return None
    return total / 1e6 if found else None


def _estimate_psutil_joules() -> float:
    """Fallback: integrate CPU utilisation × 65W TDP."""
    global _cpu_ut_prev, _cpu_wall_prev
    TDP_W = 65.0
    try:
        import psutil
        now = time.perf_counter()
        t = psutil.cpu_times()
        ut = t.user + t.system
        if _cpu_ut_prev is None or _cpu_wall_prev is None:
            _cpu_ut_prev = ut
            _cpu_wall_prev = now
            return 0.0
        wall = max(0.0, now - _cpu_wall_prev)
        cpu_d = max(0.0, ut - _cpu_ut_prev)
        _cpu_ut_prev = ut
        _cpu_wall_prev = now
        frac = min(1.0, cpu_d / max(wall, 1e-9))
        return TDP_W * wall * frac
    except Exception:
        return 0.0


def measure_cpu_energy_joules() -> float:
    """Return CPU energy (J) consumed since the last call.

    Primary source: Intel RAPL ``/sys/class/powercap/intel-rapl/*/energy_uj``.
    Falls back to psutil CPU-time × 65W TDP when RAPL is unavailable.
    """
    global _rapl_prev
    current = _read_rapl_joules()
    if current is None:
        return _estimate_psutil_joules()

    if _rapl_prev is None:
        _rapl_prev = current
        return 0.0
    # Handle 64-bit wrap-around (extremely unlikely in practice)
    if current < _rapl_prev:
        delta = current
    else:
        delta = current - _rapl_prev
    _rapl_prev = current
    return max(0.0, delta)


class EnergyMonitor:
    """Context manager that records CPU energy & time deltas.

    Usage::

        with EnergyMonitor() as mon:
            model(inputs)
        print(mon.report(tokens=total_tokens))
    """

    def __init__(self) -> None:
        self._start_j: float = 0.0
        self._end_j: float = 0.0
        self._start_t: float = 0.0
        self._end_t: float = 0.0
        self._recorded: bool = False

    def __enter__(self) -> EnergyMonitor:
        # Flush any residual reading & establish baseline
        measure_cpu_energy_joules()
        self._start_t = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end_j = measure_cpu_energy_joules()
        self._end_t = time.perf_counter()
        self._recorded = True

    def report(self, tokens: int = 0) -> Dict[str, float]:
        """Return energy summary dict.

        Keys:
            e_dyn_j          – measured dynamic energy (J)
            landauer_j       – Landauer lower bound ≈ tokens × 10³ × kT ln2
            ratio            – e_dyn_j / landauer_j
            elapsed_s        – wall-clock delta
            e_dyn_per_token  – energy per token
        """
        if not self._recorded:
            return {}
        e_dyn = max(0.0, self._end_j)
        elapsed = max(0.0, self._end_t - self._start_t)
        n_bits = max(tokens, 1) * 1000
        landauer = float(n_bits * KB * DEFAULT_T_K * LN2)
        ratio = e_dyn / landauer if landauer > 0 else float("inf")
        return {
            "e_dyn_j": e_dyn,
            "landauer_j": landauer,
            "ratio": ratio,
            "elapsed_s": elapsed,
            "e_dyn_per_token": e_dyn / max(tokens, 1),
        }


@torch.no_grad()
def mei_calibration_report(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_len: int = 128,
) -> Dict[str, Any]:
    """MEI calibration: run inference under EnergyMonitor, produce report.

    The returned dict includes the base EnergyMonitor keys plus:

        nll              – average negative log-likelihood
        n_params         – model parameter count
        landauer_params_j – Landauer bound from parameter-size estimate
        ratio_vs_params  – e_dyn / landauer_params_j
    """
    model.eval()
    total_tokens = 0
    total_loss = 0.0

    with EnergyMonitor() as em:
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
            input_ids = enc.input_ids
            n_tok = int(input_ids.numel())
            out = model(input_ids=input_ids, attention_mask=enc.attention_mask, labels=input_ids)
            total_loss += float(out.loss) * n_tok
            total_tokens += n_tok

    report = em.report(tokens=total_tokens)
    report["nll"] = total_loss / max(total_tokens, 1)
    n_params = sum(p.numel() for p in model.parameters())
    report["n_params"] = n_params
    # Refined Landauer estimate from parameter count: 16-bit × 2 ops/param
    n_param_bits = n_params * 16 * 2
    landauer_p = float(n_param_bits * KB * DEFAULT_T_K * LN2)
    report["landauer_params_j"] = landauer_p
    report["ratio_vs_params"] = report["e_dyn_j"] / max(landauer_p, 1e-30)
    return report
