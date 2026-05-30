"""Context Encoder — estable, compacto, semántico.

Convierte el residual stream ruidoso en un latente Z estable.
Tres variantes que exploran distintas formas de extraer el "concepto activo":

  WINDOW:    promedio de últimos k hidden states + proyección aleatoria
  EMA:       media móvil exponencial + proyección aleatoria
  FOURIER:   FFT de ventana temporal, DC + primer armónico

Todas comparten: determinismo, sin entrenamiento, salida de dim fija.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.fft import rfft


def _make_projection(d_model: int, context_dim: int, seed: int = 0) -> np.ndarray:
    """Semi-orthogonal random projection matrix W: (context_dim, d_model).

    Columns are orthonormal (W @ W^T ≈ I). Preserves topology
    (Johnson-Lindenstrauss) while reducing dimension.

    Handles d_model < context_dim: projects what it can, pads with noise.
    """
    rng = np.random.default_rng(seed)
    if d_model >= context_dim:
        W = rng.standard_normal((d_model, context_dim)).astype(np.float32)
        Q, _ = np.linalg.qr(W)
        return Q.T  # (context_dim, d_model)
    else:
        # d_model < context_dim: full QR + extra noise dims
        W = rng.standard_normal((d_model, d_model)).astype(np.float32)
        Q, _ = np.linalg.qr(W)
        proj = Q.T  # (d_model, d_model)
        # Pad with random noise projections
        extra = rng.standard_normal((context_dim - d_model, d_model)).astype(np.float32)
        return np.concatenate([proj, extra], axis=0)  # (context_dim, d_model)


def _project(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project h through W, then normalize."""
    z = W @ h
    n = float(np.linalg.norm(z))
    return z / n if n > 1e-8 else z


class WindowContextEncoder:
    """Promedio móvil de ventana + proyección.

    Toma los últimos k hidden states, los promedia, proyecta a context_dim.
    Incluye señal de novedad: cuánto se desvía el token actual del promedio.

    Parámetros:
        d_model: dimensión del hidden state (GPT-2: 768)
        context_dim: dimensión del latente Z de salida
        window_size: tokens a promediar (más grande = más estable)
        novelty: incluir norma del residual como feature extra
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int = 16,
        window_size: int = 3,
        novelty: bool = True,
    ):
        self.W = _make_projection(d_model, context_dim - (1 if novelty else 0))
        self.window_size = window_size
        self.novelty = novelty
        self._history: List[np.ndarray] = []
        self.context_dim = context_dim

    def reset(self) -> None:
        self._history = []

    def encode(self, h: np.ndarray) -> np.ndarray:
        h = h.ravel().astype(np.float32)
        self._history.append(h)
        if len(self._history) > self.window_size * 2:
            self._history.pop(0)

        window = self._history[-self.window_size:]
        avg = np.mean(window, axis=0)
        avg_norm = avg / (float(np.linalg.norm(avg)) + 1e-8)

        Z = _project(avg_norm, self.W)

        if self.novelty:
            novelty = float(np.linalg.norm(h - avg)) / (float(np.linalg.norm(avg)) + 1e-8)
            novelty = min(novelty, 2.0) / 2.0  # normalize to [0, 1]
            Z = np.concatenate([Z, np.array([novelty], dtype=np.float32)])

        assert len(Z) == self.context_dim, f"Z dim {len(Z)} != {self.context_dim}"
        return Z.astype(np.float32)


class EMAContextEncoder:
    """Media móvil exponencial + proyección.

    Mantiene EMA del hidden state. El rate alpha controla
    cuánto peso tiene el token actual vs el histórico.

    Bajo alpha (< 0.2) → muy estable, lento en responder.
    Alto alpha (> 0.5) → responde rápido, menos estable.

    Incluye residual signal: diferencia entre h_actual y EMA.
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int = 16,
        alpha: float = 0.3,
        residual: bool = True,
    ):
        z_dim = context_dim - (1 if residual else 0)
        self.W = _make_projection(d_model, z_dim)
        self.alpha = alpha
        self.residual = residual
        self.context_dim = context_dim
        self._ema: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._ema = None

    def encode(self, h: np.ndarray) -> np.ndarray:
        h = h.ravel().astype(np.float32)
        if self._ema is None:
            self._ema = h.copy()
        else:
            self._ema = self.alpha * h + (1.0 - self.alpha) * self._ema

        ema_norm = self._ema / (float(np.linalg.norm(self._ema)) + 1e-8)
        Z = _project(ema_norm, self.W)

        if self.residual:
            res = float(np.linalg.norm(h - self._ema))
            res_norm = res / (float(np.linalg.norm(self._ema)) + 1e-8)
            res_norm = min(res_norm, 2.0) / 2.0
            Z = np.concatenate([Z, np.array([res_norm], dtype=np.float32)])

        return Z.astype(np.float32)


class FourierContextEncoder:
    """FFT de ventana temporal + DC + primer armónico.

    Toma una ventana de N hidden states, aplica FFT en el eje temporal.
    Usa:
      - DC component: el promedio (estabilidad)
      - 1st harmonic: dirección de cambio (tendencia/topic shift)
      - High-freq energy: ruido/novedad (como señal de incertidumbre)

    Esto separa naturalmente señal estable de dinámica de ruido.
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int = 16,
        window_size: int = 4,
    ):
        # DC (mean) + 1st harmonic magnitude + novelty
        self.W_dc = _make_projection(d_model, context_dim - 2, seed=1)
        self.window_size = window_size
        self.context_dim = context_dim
        self._history: List[np.ndarray] = []

    def reset(self) -> None:
        self._history = []

    def encode(self, h: np.ndarray) -> np.ndarray:
        h = h.ravel().astype(np.float32)
        self._history.append(h)
        if len(self._history) > self.window_size * 2:
            self._history.pop(1)  # keep first element for window continuity

        window = np.stack(self._history[-self.window_size:])  # (N, d_model)
        N = window.shape[0]

        # FFT along time axis
        freqs = rfft(window, axis=0)  # (N//2+1, d_model)

        # DC (average direction)
        dc = np.real(freqs[0]) / N  # mean
        dc_norm = dc / (float(np.linalg.norm(dc)) + 1e-8)

        # 1st harmonic (trend)
        if N > 2:
            h1 = np.abs(freqs[1])  # magnitude of first harmonic
            trend_mag = float(np.mean(h1)) / (float(np.mean(np.abs(dc))) + 1e-8)
        else:
            trend_mag = 0.0

        # High-frequency energy (noise)
        if N > 3:
            hf_energy = float(np.mean(np.abs(freqs[2:]))) / (float(np.mean(np.abs(freqs[1:2]))) + 1e-8)
        else:
            hf_energy = 0.0

        # Project DC to context space
        Z = _project(dc_norm, self.W_dc)

        # Append trend and noise as scalar features
        Z = np.concatenate([
            Z,
            np.array([min(trend_mag, 2.0) / 2.0], dtype=np.float32),
            np.array([min(hf_energy, 1.0)], dtype=np.float32),
        ])

        assert len(Z) == self.context_dim, f"Z dim {len(Z)} != {self.context_dim}"
        return Z.astype(np.float32)
