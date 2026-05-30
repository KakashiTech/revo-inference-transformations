"""FieldSelector — Campo Φ/A/C que decide qué módulos modificar por token.

Φ (Field): Z codifica el contexto cognitivo
A (Activation): qué módulos (capas/heads) activar para cada token
C (Capacity): con qué intensidad modificar cada módulo seleccionado

El selector descubre automáticamente los módulos lineales del modelo,
asigna a cada uno un embedding aprendible, y mapea Z → (score, intensidad)
por módulo usando un MLP compartido con embeddings de módulo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class FieldSelectorConfig:
    context_dim: int = 16
    embed_dim: int = 8
    hidden_dim: int = 64
    score_threshold: float = 0.0
    max_active_modules: int = 0
    intensity_bias: float = 0.2
    seed: int = 0

    # Module discovery
    include_patterns: Tuple[str, ...] = (
        "h.",
        "lm_head",
    )
    exclude_patterns: Tuple[str, ...] = (
        "wte", "wpe", "ln_",
    )


_ModuleSpec = Tuple[str, nn.Module, torch.Tensor, int, int]


def discover_modules(model: nn.Module,
                     include_patterns: Tuple[str, ...],
                     exclude_patterns: Tuple[str, ...]) -> List[_ModuleSpec]:
    modules = []
    seen = set()
    for name, m in model.named_modules():
        if not hasattr(m, "weight") or not isinstance(m.weight, torch.Tensor):
            continue
        if m.weight.dim() != 2:
            continue
        if not any(p in name for p in include_patterns):
            continue
        if any(p in name for p in exclude_patterns):
            continue
        if id(m) in seen:
            continue
        seen.add(id(m))
        in_f, out_f = m.weight.shape[1], m.weight.shape[0]
        modules.append((name, m, m.weight, in_f, out_f))
    modules.sort(key=lambda x: x[0])
    return modules


def get_module_weight(model: nn.Module, path: str) -> torch.Tensor:
    parts = path.split(".")
    obj = model
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj.weight


def set_module_weight(model: nn.Module, path: str, new_weight: torch.Tensor) -> None:
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    last = parts[-1]
    target = obj if last.isdigit() else getattr(obj, last)
    target.weight.data = new_weight


class FieldSelector:
    """Mapea Z → scores e intensidades por módulo.

    Arquitectura:
        score_net: concat(Z, module_embed) → hidden → score
        intensity_net: concat(Z, module_embed) → hidden → sigmoid → [0, 1]
    """

    def __init__(self, model: nn.Module, cfg: FieldSelectorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.modules = discover_modules(
            model, cfg.include_patterns, cfg.exclude_patterns
        )
        self._n = len(self.modules)

        if self._n == 0:
            return

        d = cfg.context_dim
        e = cfg.embed_dim
        h = cfg.hidden_dim
        scale = 1.0 / max(4.0, math.sqrt(d + e))

        self.module_embed = self.rng.standard_normal((self._n, e)).astype(np.float32) * 0.1

        self.W_s1 = self.rng.standard_normal((h, d + e)).astype(np.float32) * scale
        self.b_s1 = self.rng.standard_normal((h,)).astype(np.float32) * 0.05
        self.W_s2 = self.rng.standard_normal((1, h)).astype(np.float32) * (1.0 / max(4.0, math.sqrt(h)))
        self.b_s2 = np.zeros(1, dtype=np.float32)

        hi = max(16, h // 2)
        self.W_i1 = self.rng.standard_normal((hi, d + e)).astype(np.float32) * scale
        self.b_i1 = self.rng.standard_normal((hi,)).astype(np.float32) * 0.05
        self.W_i2 = self.rng.standard_normal((1, hi)).astype(np.float32) * (1.0 / max(4.0, math.sqrt(hi)))
        self.b_i2 = np.zeros(1, dtype=np.float32)

        self._step = 0

    @property
    def n_modules(self) -> int:
        return self._n

    def select(self, Z: np.ndarray) -> List[Tuple[int, str, float, float]]:
        """Dado Z, retorna módulos seleccionados con intensidad.

        Returns:
            Lista de (module_idx, module_name, score, intensity)
        """
        if self._n == 0:
            return []

        cfg = self.cfg
        Z = Z.ravel().astype(np.float32)
        results = []

        for i in range(self._n):
            emb = self.module_embed[i]
            x = np.concatenate([Z, emb]).astype(np.float32)

            h_s = np.tanh(self.W_s1 @ x + self.b_s1)
            score = float((self.W_s2 @ h_s + self.b_s2)[0])

            h_i = np.tanh(self.W_i1 @ x + self.b_i1)
            raw_int = float((self.W_i2 @ h_i + self.b_i2)[0])
            intensity = float(1.0 / (1.0 + math.exp(-raw_int + cfg.intensity_bias)))

            results.append((i, self.modules[i][0], score, intensity))

        max_active = cfg.max_active_modules if cfg.max_active_modules > 0 else self._n
        scores_arr = np.array([r[2] for r in results])
        order = np.argsort(-scores_arr)
        selected = []
        for idx in order:
            i, name, score, intensity = results[int(idx)]
            if score > cfg.score_threshold and len(selected) < max_active:
                selected.append((int(idx), name, score, intensity))

        self._step += 1
        return selected

    def get_active_distribution(self, Z: np.ndarray) -> Dict[str, Any]:
        selected = self.select(Z)
        layer_counts: Dict[str, int] = {}
        for _, name, score, intensity in selected:
            if ".h." in name:
                parts = name.split(".h.")
                if len(parts) > 1:
                    layer_id = parts[1].split(".")[0]
                    layer_counts[f"layer_{layer_id}"] = layer_counts.get(f"layer_{layer_id}", 0) + 1
            else:
                layer_counts[name] = layer_counts.get(name, 0) + 1
        return {
            "n_selected": len(selected),
            "layer_distribution": layer_counts,
            "selected_modules": [s[1] for s in selected],
            "mean_intensity": float(np.mean([s[3] for s in selected])) if selected else 0.0,
        }

    def stats(self) -> dict:
        return {
            "n_modules": self._n,
            "module_names": [m[0] for m in self.modules],
            "max_active": self.cfg.max_active_modules,
            "threshold": self.cfg.score_threshold,
            "steps": self._step,
        }
