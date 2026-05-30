"""ActivationCache — cache de hidden states por capa para re-ejecución parcial.

Cuando el FieldSelector modifica capas intermedias, la cache permite
re-ejecutar solo desde la primera capa modificada reutilizando las
activaciones cacheadas de capas anteriores (ahorro ~67% FLOPs en GPT-2).

Uso:
    cache = ActivationCache()
    logits_base, layers = cache.cache_forward(model, input_ids)
    # ... aplicar deltas a capas 8-11 ...
    logits_mod = cache.forward_from(model, start_layer=8)
    # ... revertir deltas ...
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


def _embed(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Token + position embedding."""
    hs = model.transformer.wte(input_ids)
    hs = hs + model.transformer.wpe(
        torch.arange(input_ids.shape[1], device=input_ids.device)
    )
    return hs


class ActivationCache:
    """Cache de hidden states por capa de transformer.

    Almacena: input_ids del forward original + hidden state después
    de cada transformer block. forward_from(start_layer) reanuda
    desde start_layer usando el hidden state cacheado.
    """

    def __init__(self):
        self._cached: List[torch.Tensor] = []
        self._input_ids: Optional[torch.Tensor] = None
        self._seq_len: int = 0
        self._device: Optional[torch.device] = None

    @torch.no_grad()
    def cache_forward(self, model: nn.Module,
                      input_ids: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass completo. Cachea hidden state post-cada bloque.

        Returns:
            (logits, [h0, h1, ..., h11])
        """
        self._input_ids = input_ids.clone()
        self._seq_len = input_ids.shape[1]
        self._device = input_ids.device

        hs = _embed(model, input_ids)
        cached: List[torch.Tensor] = []
        for block in model.transformer.h:
            hs = block(hs)[0]
            cached.append(hs.clone())

        hs = model.transformer.ln_f(hs)
        logits = model.lm_head(hs)
        self._cached = cached
        return logits, cached

    @torch.no_grad()
    def forward_from(self, model: nn.Module,
                     start_layer: int) -> torch.Tensor:
        """Forward parcial desde start_layer usando activaciones cacheadas.

        Args:
            model: modelo causal LM con transformer.h
            start_layer: índice de capa desde donde re-ejecutar

        Returns:
            logits (parciales, desde start_layer en adelante)
        """
        n_total = len(model.transformer.h)
        if start_layer < 0:
            start_layer = 0
        if start_layer >= n_total:
            hs = self._cached[-1].clone() if self._cached else _embed(model, self._input_ids)
        elif start_layer == 0:
            hs = _embed(model, self._input_ids)
        else:
            hs = self._cached[start_layer - 1].clone()

        for block in list(model.transformer.h)[start_layer:]:
            hs = block(hs)[0]

        hs = model.transformer.ln_f(hs)
        logits = model.lm_head(hs)
        return logits

    def get_hidden_at(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Hidden state después de layer_idx (0-indexed)."""
        if 0 <= layer_idx < len(self._cached):
            return self._cached[layer_idx].clone()
        return None

    @property
    def n_layers(self) -> int:
        return len(self._cached)

    def clear(self) -> None:
        self._cached.clear()
        self._input_ids = None
        self._seq_len = 0

    @staticmethod
    def get_modified_range(module_names: List[str]) -> Tuple[int, int]:
        """(first_layer, last_layer) modificados, o (0,0) si solo lm_head."""
        first, last = 999, -1
        for name in module_names:
            if "lm_head" in name:
                continue
            parts = name.split(".h.")
            if len(parts) < 2:
                continue
            layer_str = parts[1].split(".")[0]
            try:
                layer = int(layer_str)
                if layer < first:
                    first = layer
                if layer > last:
                    last = layer
            except ValueError:
                continue
        if first > last:
            return (0, 0)
        return (first, last)
