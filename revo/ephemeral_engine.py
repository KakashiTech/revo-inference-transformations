"""EphemeralEngine — orquestador completo del ciclo efímero REVO.

Integra:
  - WindowContextEncoder para encoding estable de contexto
  - FieldSelector (Φ/A/C) que decide qué módulos activar por token
  - Selective gating (novedad, entropía, historial de deltas)
  - ModeCache con búsqueda por similitud Z
  - ResonantCodebook (reemplazo adaptativo de HyperLoRA)
  - Apply/revert seguro

Ciclo por token:
   1. hidden_state → WindowContextEncoder → Z
   2. FieldSelector.select(Z) → módulos activos + intensidades
   3. Buscar Z en ModeCache (exact match)
   4. Generar deltas por módulo vía codebook/HyperLoRA
   5. Selective gating: ajusta escala por módulo
   6. Apply deltas → forward → revert
   7. record_nll_delta → codebook.consolidate() si improvement
   8. Logging de métricas
"""

from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from revo._logging import get_logger
from revo.context_encoder import WindowContextEncoder
from revo.engine import apply_delta, revert_delta, DeltaHandle
from revo.hyperlora import HyperLoraConfig, hyperlora_generate
from revo.mode_cache import ModeCache
from revo.codebook_resonant import ResonantCodebook, ResonantCodebookConfig
from revo.field_selector import FieldSelector, FieldSelectorConfig
from revo.activation_cache import ActivationCache
from revo.beds import BEDSHomeostat
from revo.tqft import TopologicalProtector, TQFTConfig
from revo.oscillatory_gating import OscillatoryHooks
from revo.holography import replace_with_holography
from revo.wdm import replace_with_wdm
from revo.primitiva_router import PrimitiveModel as _PrimitiveModel
from revo.phase_bus import replace_with_phase_bus
from revo.reversible import replace_with_reversible
from revo.holomorphic import replace_mlp_with_holomorphic
from revo.hora import replace_with_hora
from revo.spectral import prune_model_spectral
from revo.fft_kernel import replace_with_circulant

log = get_logger(__name__)


@dataclass
class EphemeralConfig:
    # Context encoder
    context_dim: int = 16
    window_size: int = 4

    # HyperLoRA (fallback when codebook disabled)
    rank: int = 4
    scale_min: float = 0.0  # 0 = allow complete skip via gating
    scale_max: float = 0.8   # conservative upper bound
    seed: int = 0

    # ResonantCodebook (replaces HyperLoRA when enabled)
    use_codebook: bool = True
    codebook_k: int = 64
    codebook_max_k: int = 256
    codebook_lr: float = 0.15
    codebook_top_k: int = 3
    codebook_exploration: float = 0.04
    codebook_add_threshold: float = 0.7
    codebook_seed_from_hyperlora: int = 0  # >0: seed first N tokens' Z's via HyperLoRA

    # FieldSelector (Φ/A/C)
    use_field_selector: bool = True
    field_max_active: int = 0  # 0 = all modules eligible
    field_threshold: float = 0.0
    field_intensity_bias: float = 0.2

    # ModeCache
    use_cache: bool = True
    cache_ttl: int = 3600
    cache_max: int = 128
    cache_similarity: float = 0.92

    # Selective gating
    # Novelty: skip when topic shift detected
    novelty_threshold: float = 0.3
    # Running delta window: if recent NLL deltas are positive, reduce scale
    delta_window: int = 3
    delta_window_threshold: float = 0.02
    delta_window_scale_factor: float = 0.3
    # Max history length for novelty computation
    max_novelty_history: int = 20

    # Runtime enhancements
    use_beds: bool = False
    beds_h_min: float = 0.5
    beds_h_max: float = 3.5
    beds_kp: float = 0.05
    beds_ki: float = 0.01
    beds_window: int = 10

    use_tqft: bool = False
    tqft_topk: int = 64
    tqft_num_braids: int = 5
    tqft_noise_sigma: float = 0.05

    use_oscillatory: bool = False
    osc_alpha: float = 0.05
    osc_freq: float = 0.25

    # Primitive Router (token-conditional computational primitive selection)
    use_primitiva_router: bool = False
    primitiva_enabled: Optional[List[str]] = None  # None = all available
    primitiva_router_hard: bool = True
    primitiva_router_temperature: float = 1.0

    # Weight-replacement modules (applied once at init)
    use_holography: bool = False
    holography_boundary_dim: int = 64
    holography_alpha: float = 1.0

    use_wdm: bool = False
    wdm_bands: int = 2

    use_phase_bus: bool = False

    use_reversible: bool = False
    reversible_rank: int = 2

    use_holomorphic: bool = False
    holomorphic_rank: int = 8

    use_hora: bool = False
    hora_rank: int = 4
    hora_alpha: float = 1.0
    hora_c: float = 0.0

    use_spectral_pruning: bool = False
    spectral_energy_keep: float = 0.9

    use_circulant: bool = False


class EphemeralEngine:
    """Orquestador del ciclo efímero completo.

    Integra todos los módulos REVO: HyperLoRA, FieldSelector, Codebook,
    ModeCache, ActivationCache, BEDS, TQFT, OscillatoryHooks, y más.

    Uso:
        engine = EphemeralEngine(model, config)
        engine.reset()  # al inicio de cada secuencia
        logits, meta = engine.step(hidden_state, token_idx)
    """

    def __init__(self, model: nn.Module, cfg: Optional[EphemeralConfig] = None):
        self.cfg = cfg or EphemeralConfig()
        self.model = model
        self.device = next(model.parameters()).device

        # lm_head
        self.lm_head = model.lm_head if hasattr(model, "lm_head") else model.transformer.wte
        self.out_f, self.in_f = self.lm_head.weight.shape

        # Context encoder — use actual lm_head input dimension
        d_model = self.lm_head.in_features
        self.encoder = WindowContextEncoder(
            d_model=d_model,
            context_dim=self.cfg.context_dim,
            window_size=self.cfg.window_size,
            novelty=True,
        )

        # HyperLoRA config (fallback)
        self.hl_cfg = HyperLoraConfig(
            context_dim=self.cfg.context_dim,
            rank=self.cfg.rank,
            in_features=self.in_f,
            out_features=self.out_f,
            scale_min=self.cfg.scale_min,
            scale_max=self.cfg.scale_max,
            seed=self.cfg.seed,
        )

        # FieldSelector (Φ/A/C)
        self.field: Optional[FieldSelector] = None
        if self.cfg.use_field_selector:
            fs_cfg = FieldSelectorConfig(
                context_dim=self.cfg.context_dim,
                score_threshold=self.cfg.field_threshold,
                max_active_modules=self.cfg.field_max_active,
                intensity_bias=self.cfg.field_intensity_bias,
                seed=self.cfg.seed,
            )
            self.field = FieldSelector(model, fs_cfg)

        # Discover transformer blocks for multi-layer support
        self._transformer_blocks: List[nn.Module] = []
        self._has_blocks: bool = False
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self._transformer_blocks = list(model.transformer.h)
            self._has_blocks = len(self._transformer_blocks) > 0
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self._transformer_blocks = list(model.model.layers)
            self._has_blocks = len(self._transformer_blocks) > 0

        # ResonantCodebook
        self.codebook: Optional[ResonantCodebook] = None
        self._cb_warmup_buffer: List[np.ndarray] = []
        self._cb_warmup_deltas: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self._cb_warmup_done: bool = False
        if self.cfg.use_codebook:
            cb_cfg = ResonantCodebookConfig(
                context_dim=self.cfg.context_dim,
                rank=self.cfg.rank,
                in_features=self.in_f,
                out_features=self.out_f,
                k=self.cfg.codebook_k,
                max_k=self.cfg.codebook_max_k,
                lr=self.cfg.codebook_lr,
                top_k=self.cfg.codebook_top_k,
                exploration_scale=self.cfg.codebook_exploration,
                add_threshold=self.cfg.codebook_add_threshold,
                scale_min=self.cfg.scale_min,
                scale_max=self.cfg.scale_max,
                seed=self.cfg.seed,
            )
            self.codebook = ResonantCodebook(cb_cfg)

        # ModeCache
        self.cache: Optional[ModeCache] = None
        if self.cfg.use_cache:
            self.cache = ModeCache(
                ttl_seconds=self.cfg.cache_ttl,
                max_size=self.cfg.cache_max,
                similarity_threshold=self.cfg.cache_similarity,
            )

        # ActivationCache (para re-ejecución parcial multi-capa)
        self.act_cache = ActivationCache()

        # BEDS homeostat (post-procesamiento de logits)
        self.beds: Optional[BEDSHomeostat] = None

        # TQFT topological protection (post-procesamiento de logprobs)
        self.tqft: Optional[TopologicalProtector] = None

        # Oscillatory gating hooks
        self.osc: Optional[OscillatoryHooks] = None

        # Initialize runtime enhancement modules
        self._init_enhancements()

        # PrimitiveModel — wraps Linears with token-conditional primitive selection
        self.primitive_model = None
        if self.cfg.use_primitiva_router:
            log.info("Applying PrimitiveModel (primitives=%s, hard=%s)...",
                     self.cfg.primitiva_enabled or "all", self.cfg.primitiva_router_hard)
            self.primitive_model = _PrimitiveModel(
                self.model,
                enabled=self.cfg.primitiva_enabled,
                router_hard=self.cfg.primitiva_router_hard,
                router_temperature=self.cfg.primitiva_router_temperature,
            )
            log.info("  → replaced %d layers with PrimitiveSelectors",
                     self.primitive_model.describe()["replaced_layers"])

        # Apply weight-replacement modules (in-place model transformations)
        self._replacements_report: Dict[str, Any] = {}
        self._apply_weight_replacements()

        # State
        self.reset()

    def reset(self) -> None:
        """Reset per-sequence state + lifetime counters."""
        self.reset_sequence()
        self._n_steps = 0
        self._total_delta_nll = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_similar_hits = 0
        self._gated_skip = 0
        self._gated_reduced = 0
        self._step_times = []
        self._codebook_contribs = 0
        self._codebook_additions = 0
        self._field_calls = 0
        self._field_activations: List[int] = []
        self._cb_warmup_buffer = []
        self._cb_warmup_deltas = []
        self._cb_warmup_done = False

    def reset_sequence(self) -> None:
        """Reset per-sequence state only (encoder + gating window)."""
        self.encoder.reset()
        self._recent_deltas = []
        self._last_contrib_codes = None
        self._last_contrib_weights = None
        self._last_Z = None
        self._last_A = None
        self._last_B = None
        self._last_scale = None
        self._last_field_modules = []

    # -- Selective gating --

    def _gate(self, scale: float, novelty: float, entropy: Optional[float] = None) -> Tuple[float, str]:
        """Apply selective gating. Returns (scale, reason)."""
        if novelty > self.cfg.novelty_threshold:
            return 0.0, "novelty"
        if len(self._recent_deltas) >= self.cfg.delta_window:
            recent_mean = np.mean(self._recent_deltas[-self.cfg.delta_window:])
            if recent_mean > self.cfg.delta_window_threshold:
                scale *= self.cfg.delta_window_scale_factor
                if scale < 0.01:
                    return 0.0, "delta_window"
                return scale, f"delta_window({recent_mean:.3f})"
        return scale, ""

    # -- Field-based delta generation --

    def _generate_delta(self, Z: np.ndarray,
                        module_name: str,
                        in_features: int,
                        out_features: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Generate (A, B, scale) for a specific module using codebook or HyperLoRA."""
        if self.codebook is not None:
            warmup_n = self.cfg.codebook_seed_from_hyperlora
            if warmup_n > 0 and not self._cb_warmup_done and len(self._cb_warmup_buffer) < warmup_n:
                A, B, scale = hyperlora_generate(Z, self.hl_cfg)
                self._cb_warmup_buffer.append(Z.copy())
                self._cb_warmup_deltas.append((A, B, scale))
                if len(self._cb_warmup_buffer) >= warmup_n:
                    self.codebook.seed_from_hyperlora(self._cb_warmup_buffer, self.hl_cfg)
                    self._cb_warmup_buffer = []
                    self._cb_warmup_deltas = []
                    self._cb_warmup_done = True
                return A, B, scale
            A, B, scale, contrib, weights = self.codebook.query(Z)
            self._last_contrib_codes = contrib
            self._last_contrib_weights = weights
            self._codebook_contribs += 1
            return A, B, scale
        hl_cfg = HyperLoraConfig(
            context_dim=self.cfg.context_dim,
            rank=self.cfg.rank,
            in_features=in_features,
            out_features=out_features,
            scale_min=self.cfg.scale_min,
            scale_max=self.cfg.scale_max,
            seed=self.cfg.seed,
        )
        return hyperlora_generate(Z, hl_cfg)

    # -- Main step (lm_head only, fast path) --

    @torch.no_grad()
    def step(self, hidden_state: torch.Tensor, token_idx: int,
             entropy: Optional[float] = None) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Un paso del ciclo efímero sobre lm_head.

        Si el field selector está activo, modula la escala según la intensidad
        del campo para lm_head. Selecciona módulos adicionales pero solo
        modifica lm_head en este path rápido.

        Args:
            hidden_state: residual stream en posición token_idx (antes de lm_head)
            token_idx: posición en la secuencia

        Returns:
            (logits_modificados, metadata) o (None, metadata) si se saltea
        """
        t0 = time.perf_counter()
        h = hidden_state.detach().float().cpu().numpy().ravel().astype(np.float32)
        meta: Dict[str, Any] = {"step": self._n_steps, "token_idx": token_idx}

        # 1. Encode context → Z
        Z = self.encoder.encode(h)
        novelty = Z[-1]
        meta["z_norm"] = float(np.linalg.norm(Z))
        meta["novelty"] = float(novelty)

        # 2. Field selector (Φ/A/C)
        field_intensity = 1.0
        self._last_field_modules = []
        if self.field is not None:
            self._field_calls += 1
            selected = self.field.select(Z)
            lm_head_selected = [s for s in selected if "lm_head" in s[1]]
            other_selected = [s for s in selected if "lm_head" not in s[1]]
            self._last_field_modules = selected

            if lm_head_selected:
                field_intensity = float(lm_head_selected[0][3])
                field_intensity = max(0.0, min(1.0, field_intensity))
            meta["field_n_selected"] = len(selected)
            meta["field_intensity"] = field_intensity
            if other_selected:
                meta["field_other_modules"] = [s[1] for s in other_selected[:4]]
                self._field_activations.append(len(other_selected))

        # 3. ModeCache lookup
        cached_delta = None
        cache_key = f"w{self.cfg.window_size}_{hash(Z.tobytes()) & 0xFFFFFFFF:08x}"
        if self.cache is not None:
            if self.codebook is not None:
                exact = self.cache.get(cache_key)
                if exact is not None:
                    A, B, scale = exact
                    cached_delta = (A, B, scale, "exact")
                    self._cache_hits += 1
                else:
                    self._cache_misses += 1
            else:
                exact = self.cache.get(cache_key)
                if exact is not None:
                    A, B, scale = exact
                    cached_delta = (A, B, scale, "exact")
                    self._cache_hits += 1
                else:
                    similar, sim_key, sim_val = self.cache.get_similar(Z)
                    if similar is not None:
                        A, B, scale = similar
                        cached_delta = (A, B, scale, f"sim({sim_val:.3f})")
                        self._cache_similar_hits += 1
                    else:
                        self._cache_misses += 1

        # 4. Generate or retrieve delta for lm_head
        self._last_contrib_codes = None
        self._last_contrib_weights = None
        self._last_Z = Z.copy()

        if cached_delta is not None:
            A, B, scale, source = cached_delta
            meta["cache"] = source
        else:
            A, B, scale = self._generate_delta(Z, "lm_head", self.in_f, self.out_f)
            if self.codebook is not None:
                meta["cache"] = "codebook"
            else:
                meta["cache"] = "miss"

        self._last_A = A.copy()
        self._last_B = B.copy()
        self._last_scale = float(scale)

        # 5. Apply field intensity + scale bounds
        scale = scale * field_intensity
        scale = max(self.cfg.scale_min, min(self.cfg.scale_max, scale))
        meta["base_scale"] = float(scale)

        # 6. Selective gating
        scale, gate_reason = self._gate(scale, float(novelty), entropy)
        meta["gate_reason"] = gate_reason
        meta["final_scale"] = float(scale)

        if scale < 1e-6:
            self._gated_skip += 1
            self._step_times.append((time.perf_counter() - t0) * 1000)
            return None, {**meta, "skipped": True}

        if gate_reason:
            self._gated_reduced += 1

        # 7. Apply delta
        W_np = self.lm_head.weight.detach().float().cpu().numpy()
        W_new, handle = apply_delta(W_np, A, B, scale)
        self.lm_head.weight.data = torch.from_numpy(W_new).to(
            device=self.device, dtype=self.lm_head.weight.dtype
        )

        # 8. Forward pass
        logits = self.lm_head(hidden_state.to(dtype=self.lm_head.weight.dtype))

        # 9. Revert delta
        W_revert = revert_delta(W_new, handle)
        self.lm_head.weight.data = torch.from_numpy(W_revert).to(
            device=self.device, dtype=self.lm_head.weight.dtype
        )

        # 10. Cache delta
        if self.cache is not None and cached_delta is None:
            self.cache.put(cache_key, (A, B, scale), z=Z)

        # 11. Post-process logits (BEDS, TQFT)
        logits = self.post_process_logits(logits)

        # 12. Update state
        self._n_steps += 1
        t_ms = (time.perf_counter() - t0) * 1000
        self._step_times.append(t_ms)
        meta["step_time_ms"] = t_ms

        return logits, {**meta, "skipped": False}

    # -- Full step with multi-layer modification --

    @torch.no_grad()
    def step_full(self, input_ids: torch.Tensor, position: int,
                  use_cache: bool = True) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Ciclo efímero completo con modificación multi-capa + activation cache.

        Toma input_ids, corre forward con ActivationCache para cachear
        hidden states por capa. FieldSelector decide qué módulos modificar.
        Se re-ejecuta solo desde la primera capa modificada (ahorro ~67% FLOPs).

        Args:
            input_ids: (1, seq_len) tensor de tokens
            position: posición del token actual
            use_cache: usar activation cache (default True)

        Returns:
            (logits_modificados, metadata)
        """
        t0 = time.perf_counter()
        meta: Dict[str, Any] = {"step": self._n_steps, "mode": "full", "cached": use_cache}

        # 0. Forward pass completo con cache de activaciones
        logits_base, layer_hidden = self.act_cache.cache_forward(self.model, input_ids)
        h_t = layer_hidden[-1][0, position] if layer_hidden else logits_base[0, position]

        h = h_t.detach().float().cpu().numpy().ravel().astype(np.float32)

        # 1. Encode context → Z
        Z = self.encoder.encode(h)
        novelty = Z[-1]
        meta["z_norm"] = float(np.linalg.norm(Z))
        meta["novelty"] = float(novelty)

        # 2. Field selector
        field_intensity = 1.0
        selected_modules: List[Tuple[int, str, float, float]] = []
        if self.field is not None:
            self._field_calls += 1
            selected_modules = self.field.select(Z)
            meta["field_n_selected"] = len(selected_modules)
            lm_head_sel = [s for s in selected_modules if "lm_head" in s[1]]
            if lm_head_sel:
                field_intensity = max(0.0, min(1.0, float(lm_head_sel[0][3])))
                meta["field_intensity"] = field_intensity

        # 3. Gate
        base_scale = max(0.1, min(self.cfg.scale_max, field_intensity))
        scale, gate_reason = self._gate(base_scale, float(novelty))
        meta["gate_reason"] = gate_reason

        if scale < 1e-6:
            self._gated_skip += 1
            self._step_times.append((time.perf_counter() - t0) * 1000)
            return None, {**meta, "skipped": True}

        # 4. Generate deltas for selected modules
        mod_names_sel = [s[1] for s in selected_modules]
        deltas: List[Tuple[int, nn.Module, np.ndarray, np.ndarray, float]] = []
        for mod_idx, mod_name, score, intensity in selected_modules:
            if mod_idx >= len(self.field.modules):
                continue
            _, mod_w, W, in_f, out_f = self.field.modules[mod_idx]
            if "lm_head" in mod_name:
                A, B, s = self._generate_delta(Z, mod_name, self.in_f, self.out_f)
                s = s * field_intensity
            else:
                A, B, s = self._generate_delta(Z, mod_name, in_f, out_f)
                s = s * float(intensity)
            s = max(self.cfg.scale_min, min(self.cfg.scale_max, s))
            deltas.append((mod_idx, mod_w, A, B, s))

        if not deltas:
            self._gated_skip += 1
            self._step_times.append((time.perf_counter() - t0) * 1000)
            return None, {**meta, "skipped": True}

        # 5. Apply all deltas
        handles: List[Tuple[int, Any]] = []
        for mod_idx, mod, A, B, s in deltas:
            W_np = mod.weight.detach().float().cpu().numpy()
            W_new, handle = apply_delta(W_np, A, B, s)
            mod.weight.data = torch.from_numpy(W_new).to(
                device=self.device, dtype=mod.weight.dtype
            )
            handles.append((mod_idx, handle))

        # 6. Re-run forward desde la primera capa modificada
        first_layer, _ = ActivationCache.get_modified_range(mod_names_sel)
        if use_cache and first_layer > 0:
            logits = self.act_cache.forward_from(self.model, start_layer=first_layer)
            meta["rerun_layers"] = f"{first_layer}-{len(self.model.transformer.h) - 1}"
        else:
            with torch.no_grad():
                hs = self.model.transformer.wte(input_ids)
                hs = hs + self.model.transformer.wpe(
                    torch.arange(input_ids.shape[1], device=self.device)
                )
                for block in self.model.transformer.h:
                    hs = block(hs)[0]
                hs = self.model.transformer.ln_f(hs)
                logits = self.model.lm_head(hs)
            meta["rerun_layers"] = "full"

        # 7. Revert all deltas
        for mod_idx, handle in handles:
            _, mod_w, _, _, _ = self.field.modules[mod_idx]
            W_cur = mod_w.weight.detach().float().cpu().numpy()
            W_rev = revert_delta(W_cur, handle)
            mod_w.weight.data = torch.from_numpy(W_rev).to(
                device=self.device, dtype=mod_w.weight.dtype
            )

        meta["rerun_first_layer"] = first_layer
        meta["n_deltas_applied"] = len(handles)

        # 8. Post-process logits (BEDS, TQFT)
        if logits is not None:
            logits_at_pos = logits[0, position]
            logits_at_pos = self.post_process_logits(logits_at_pos)
        else:
            logits_at_pos = None

        # 9. Update state
        self._n_steps += 1
        t_ms = (time.perf_counter() - t0) * 1000
        self._step_times.append(t_ms)
        meta["step_time_ms"] = t_ms

        return logits_at_pos, {**meta, "skipped": False}

    # -- Feedback --

    def record_nll_delta(self, delta: float) -> None:
        """Feed back the NLL delta for gating and codebook consolidation."""
        self._recent_deltas.append(delta)
        self._total_delta_nll += delta
        if len(self._recent_deltas) > self.cfg.max_novelty_history:
            self._recent_deltas.pop(0)

        if (self.codebook is not None and delta < -0.01
                and self._last_contrib_codes is not None
                and self._last_Z is not None
                and self._last_A is not None
                and self._last_B is not None
                and self._last_scale is not None):
            old_len = len(self.codebook.codes)
            self.codebook.consolidate(
                self._last_Z, self._last_A, self._last_B, self._last_scale, delta,
                self._last_contrib_codes, self._last_contrib_weights,
            )
            if len(self.codebook.codes) > old_len:
                self._codebook_additions += 1

    # -- Stats --

    def stats(self) -> Dict[str, Any]:
        s = {
            "steps": self._n_steps,
            "cache_hits": self._cache_hits,
            "cache_similar_hits": self._cache_similar_hits,
            "cache_misses": self._cache_misses,
            "gated_skip": self._gated_skip,
            "gated_reduced": self._gated_reduced,
            "total_nll_delta": self._total_delta_nll,
            "mean_step_time_ms": float(np.mean(self._step_times)) if self._step_times else 0.0,
        }
        if self.codebook is not None:
            s["codebook"] = self.codebook.stats()
            s["codebook_contribs"] = self._codebook_contribs
            s["codebook_additions"] = self._codebook_additions
        if self.field is not None:
            s["field"] = self.field.stats()
            s["field_calls"] = self._field_calls
            s["field_mean_other"] = float(np.mean(self._field_activations)) if self._field_activations else 0.0
        if self._replacements_report:
            s["weight_replacements"] = self._replacements_report
        if self.primitive_model is not None:
            s["primitive_router"] = self.primitive_model.describe()
        return s

    def save_log(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.stats(), f, indent=2, default=str)

    # -- Enhancement module initialization --

    def _init_enhancements(self) -> None:
        """Initialize runtime enhancement modules (BEDS, TQFT, oscillatory hooks)."""
        if self.cfg.use_beds:
            self.beds = BEDSHomeostat(
                H_min=self.cfg.beds_h_min,
                H_max=self.cfg.beds_h_max,
                Kp=self.cfg.beds_kp,
                Ki=self.cfg.beds_ki,
                window=self.cfg.beds_window,
            )
            log.info("BEDS homeostat initialized: H=[%.2f, %.2f]", self.cfg.beds_h_min, self.cfg.beds_h_max)

        if self.cfg.use_tqft:
            tqft_cfg = TQFTConfig(
                topk=self.cfg.tqft_topk,
                num_braids=self.cfg.tqft_num_braids,
                noise_sigma=self.cfg.tqft_noise_sigma,
                seed=self.cfg.seed,
            )
            self.tqft = TopologicalProtector(tqft_cfg)
            log.info("TQFT protector initialized: topk=%d braids=%d", tqft_cfg.topk, tqft_cfg.num_braids)

        if self.cfg.use_oscillatory:
            self.osc = OscillatoryHooks(
                alpha=self.cfg.osc_alpha,
                freq=self.cfg.osc_freq,
            )
            selected = self.osc.attach(self.model, skip_lm_head=True)
            log.info("Oscillatory hooks attached to %d modules", len(selected))

    def post_process_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply runtime enhancements to logits (BEDS, TQFT) after forward pass."""
        if self.beds is not None:
            logits = self.beds.correct(logits)
        if self.tqft is not None:
            logits = self.tqft.protect_logprobs(logits)
            # TQFT returns logprobs, convert back if needed
            logits = torch.log_softmax(logits, dim=-1)
        return logits

    def _apply_weight_replacements(self) -> None:
        """Apply all weight-replacement modules to the model in-place.

        Order (wrapping/low-rank before structural to preserve `weight` attr):
          1. Spectral pruning (weight modification, preserves Linear)
          2. Holography (wraps Linear with FFT harmonic transforms)
          3. PhaseBus (wraps with phase alignment)
          4. Reversible (replaces MLP with reversible layers)
          5. Holomorphic (replaces MLP with holomorphic layers)
          6. HORA (replaces Linear with HORA low-rank layers)
          7. WDM (replaces Linear with FFT-based parallel channels)
          8. Circulant (replaces Linear with circulant FFT)
        """
        cfg = self.cfg

        # 1. Spectral pruning (weight modification, keeps Linear structure)
        if cfg.use_spectral_pruning:
            log.info("Applying spectral pruning (energy_keep=%.2f)...", cfg.spectral_energy_keep)
            r = prune_model_spectral(self.model, energy_keep=cfg.spectral_energy_keep)
            self._replacements_report["spectral"] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r.items()}
            log.info("  → pruned %d layers", len(r))

        # 2. Holography — wraps Linear with FFT harmonic transforms
        if cfg.use_holography:
            log.info("Applying holography (boundary_dim=%d, alpha=%.2f)...", cfg.holography_boundary_dim, cfg.holography_alpha)
            r = replace_with_holography(self.model, boundary_dim=cfg.holography_boundary_dim, alpha=cfg.holography_alpha)
            self._replacements_report["holography"] = {k: list(v) for k, v in r.items()}
            log.info("  → wrapped %d layers", len(r))

        # 3. PhaseBus — wraps with phase alignment
        if cfg.use_phase_bus:
            log.info("Applying phase bus replacement...")
            r = replace_with_phase_bus(self.model)
            self._replacements_report["phase_bus"] = {k: list(v) for k, v in r.items()}
            log.info("  → replaced %d layers", len(r))

        # 4. Reversible — replaces MLP with reversible layers
        if cfg.use_reversible:
            log.info("Applying reversible (rank=%d)...", cfg.reversible_rank)
            r = replace_with_reversible(self.model, rank=cfg.reversible_rank)
            self._replacements_report["reversible"] = {k: list(v) for k, v in r.items()}
            log.info("  → replaced %d layers", len(r))

        # 5. Holomorphic — replaces MLP with holomorphic layers
        if cfg.use_holomorphic:
            log.info("Applying holomorphic (rank=%d)...", cfg.holomorphic_rank)
            r = replace_mlp_with_holomorphic(self.model, rank=cfg.holomorphic_rank)
            self._replacements_report["holomorphic"] = {k: round(v, 4) for k, v in r.items()}
            log.info("  → replaced %d parameters", len(r))

        # 6. HORA — replaces Linear with HORA low-rank layers
        if cfg.use_hora:
            log.info("Applying HORA (rank=%d, alpha=%.2f, c=%.2f)...", cfg.hora_rank, cfg.hora_alpha, cfg.hora_c)
            r = replace_with_hora(self.model, rank=cfg.hora_rank, alpha=cfg.hora_alpha, c=cfg.hora_c)
            self._replacements_report["hora"] = {k: list(v) for k, v in r.items()}
            log.info("  → replaced %d layers", len(r))

        # 7. WDM — replaces Linear with FFT-based parallel channels
        if cfg.use_wdm:
            log.info("Applying WDM (bands=%d)...", cfg.wdm_bands)
            r = replace_with_wdm(self.model, bands=cfg.wdm_bands)
            self._replacements_report["wdm"] = {k: list(v) for k, v in r.items()}
            log.info("  → replaced %d layers", len(r))

        # 8. Circulant — replaces Linear with circulant FFT
        if cfg.use_circulant:
            log.info("Applying circulant replacement...")
            r = replace_with_circulant(self.model)
            self._replacements_report["circulant"] = {k: v for k, v in r.items()}
            log.info("  → replaced %d layers", len(r))

        if self._replacements_report:
            log.info("Weight replacements applied: %s", list(self._replacements_report.keys()))
