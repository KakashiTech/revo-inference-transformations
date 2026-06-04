"""Law Streaming — the model IS the law, weights are ephemeral.

Native forward mode (default for inference): generates weights and assigns them
directly to block parameters, then calls the native block forward. No
``functional_call`` overhead — the same computation as the original model,
just with law-generated weight values.

Training mode (``finetune_law``): still uses ``functional_call`` to maintain
the computation graph for gradient flow through generated weights back into
the law parameters.

The core paradigm shift: instead of storing weight matrices on disk and loading
them per layer, we store a Generative Weight Law — a compact neural network
that PRODUCES the weight matrices on demand. The law is loaded once into RAM
and generates each layer's weights: generate → assign → forward → destroy.

This is the TRUE REVO:
    "El modelo no es un tensor: es una ley de transformación reproducible."

Usage:
    law = build_law(model, ...)
    logits = law_stream_forward(model, input_ids, law, use_native=True)
    tokens = law_generate(model, tokenizer, prompt, law, use_native=True)

"""

from __future__ import annotations

import gc
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from revo._utils import free_memory_trim, measure_memory_rss

from revo.streaming import (
    _detect_arch, _n_layers, _d_model, _get_layer_container,
    _arch_layer_prefix, _layer_pattern, _non_layer_pattern,
    _shared_keys, _extract_shared, _get_rotary_emb,
    _build_block_kwargs, _get_device, _rms_norm,
)


# ─── SVD Target Extraction ──────────────────────────────────────────────

@torch.no_grad()
def _extract_svd_targets(model: nn.Module, rank: int) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute SVD factors of every 2D weight in every block.

    Returns {layer_idx: {weight_base: {U_eff, Vh}}}.
    """
    state = model.state_dict()
    arch = _detect_arch(model)
    n_layers = _n_layers(model)
    targets: Dict[str, Dict[str, torch.Tensor]] = {}

    for i in range(n_layers):
        pat = _layer_pattern(i, arch)
        arch_prefix = _arch_layer_prefix(i, arch)
        prefix_len = len(arch_prefix)
        layer: Dict[str, torch.Tensor] = {}

        for key, tensor in state.items():
            if not pat.search(key) or tensor.dim() < 2:
                continue
            m, n = tensor.shape
            r = min(rank, m, n)
            relative = key[prefix_len:] if prefix_len and key.startswith(arch_prefix) else key
            base = relative[:-7] if relative.endswith('.weight') and len(relative) > 7 else relative

            W = tensor.float().cpu()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            U_eff = U[:, :r] * S[:r].unsqueeze(0)
            Vh_trunc = Vh[:r, :].contiguous()
            layer[base] = {"U_eff": U_eff, "Vh": Vh_trunc}

        targets[str(i)] = layer

    return targets


# ─── Phase XII-c: Cognitive Field & Selective Generation ──────────────

class FieldState:
    """Φ/A/C field state governing selective weight generation.

    Attributes:
        phi: valence in [-1, 1] (positive/negative bias)
        arousal: energy/exploration in [0, 1]
        coherence: stability in [0, 1]
        uncertainty: entropy-driven in [0, 1]
        scale: global magnitude multiplier in [0, 1]
        layer_scores: per-layer relevance in [0, 1] (n_layers,)
    """
    def __init__(self, phi: torch.Tensor, arousal: torch.Tensor,
                 coherence: torch.Tensor, uncertainty: torch.Tensor,
                 scale: torch.Tensor, layer_scores: torch.Tensor):
        self.phi = phi
        self.arousal = arousal
        self.coherence = coherence
        self.uncertainty = uncertainty
        self.scale = scale
        self._layer_scores = layer_scores

    def layer_score(self, layer_idx: int) -> torch.Tensor:
        """Relevance score for a specific layer (0-dim tensor, keeps grad flow)."""
        return self._layer_scores[layer_idx]

    def layers_to_generate(self, threshold: float = 0.3) -> List[int]:
        """List of layer indices whose score exceeds threshold."""
        return [i for i, s in enumerate(self._layer_scores)
                if s.item() > threshold]

    def __repr__(self) -> str:
        return (f"Φ={self.phi.item():.3f} A={self.arousal.item():.3f} "
                f"C={self.coherence.item():.3f} U={self.uncertainty.item():.3f} "
                f"scale={self.scale.item():.3f} "
                f"active_layers={len(self.layers_to_generate())}/{len(self._layer_scores)}")


class CognitiveField(nn.Module):
    """The Φ/A/C cognitive field.

    Computes valence, arousal, coherence, uncertainty, and per-layer
    relevance scores from the hidden state. Governs what to reconstruct,
    with what magnitude, and at what speed — the executive control
    layer of REVO Phase XII-c.

    Layer scores follow a heuristic: deeper layers are more
    task-specific and thus more likely to need regeneration. The
    arousal axis scales how many layers are active — high arousal
    (exploratory) activates more layers, low arousal (conservative)
    activates fewer.

    Architecture:
        hidden_state → Linear(4) → Φ, A, C, U
                      → ScaleNet → global scale
                      → layer_depth * arousal → per-layer relevance
    """

    def __init__(self, d_model: int, n_layers: int, hidden_dim: int = 64):
        super().__init__()
        self.field_proj = nn.Linear(d_model, 4)
        self.scale_net = nn.Linear(4, 1)
        # Heuristic depth bias: deeper layers get higher baseline relevance
        self.register_buffer('layer_depth',
                              torch.linspace(0.15, 1.0, n_layers))

    def forward(self, hidden_state: torch.Tensor) -> FieldState:
        pooled = hidden_state.mean(dim=1)  # (B, d_model)
        raw = self.field_proj(pooled)  # (B, 4)
        phi = torch.tanh(raw[:, 0])
        arousal = torch.sigmoid(raw[:, 1])
        coherence = torch.sigmoid(raw[:, 2])
        uncertainty = torch.sigmoid(raw[:, 3])
        scale = torch.sigmoid(self.scale_net(raw)).squeeze(-1)
        # Layer relevance: deep layers always matter; arousal gates
        # how many shallow layers are also regenerated.
        layer_scores = self.layer_depth * (0.4 + 0.6 * arousal)
        return FieldState(phi, arousal, coherence, uncertainty,
                          scale, layer_scores)


class WeightModeCache:
    """Cache of generated weights keyed by quantized field state.

    When the field state is similar to a previous one, weights are
    reused instead of regenerated — the "resonance as cache" principle
    from REVO Phase 3.

    The cache key is a discretized tuple of (φ, α, κ, у) at ``precision``
    bits each. LRU eviction when ``max_size`` is exceeded.
    """

    def __init__(self, max_size: int = 64, precision: int = 4):
        self.cache: Dict[Tuple, Dict[int, Dict]] = {}
        self.max_size = max_size
        self.precision = precision
        self._order: List[Tuple] = []  # LRU tracking

    def _make_key(self, field_state: FieldState) -> Tuple:
        p = self.precision
        bits = (
            int(field_state.phi.item() * (2 ** (p - 1) - 1)),
            int(field_state.arousal.item() * (2 ** p - 1)),
            int(field_state.coherence.item() * (2 ** p - 1)),
            int(field_state.uncertainty.item() * (2 ** p - 1)),
        )
        return bits

    def get(self, field_state: FieldState, layer_idx: int):
        key = self._make_key(field_state)
        if key in self.cache and layer_idx in self.cache[key]:
            self._order.remove(key)
            self._order.append(key)
            return self.cache[key][layer_idx]
        return None

    def put(self, field_state: FieldState, layer_idx: int, weights):
        key = self._make_key(field_state)
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                oldest = self._order.pop(0)
                del self.cache[oldest]
            self.cache[key] = {}
            self._order.append(key)
        self.cache[key][layer_idx] = weights

    def __len__(self) -> int:
        return len(self.cache)


# ─── Generative Weight Law ──────────────────────────────────────────────

class OutputHead(nn.Module):
    """Generates SVD factors for one weight matrix via factorized bases.

    Instead of generating U_eff (out, r) and Vh (r, in) directly (which
    requires a huge output projection), we generate SMALL coefficient
    matrices that combine with LEARNED bases:

        U_eff = U_basis @ U_coeff     (out, r) = (out, s) @ (s, r)
        Vh    = V_coeff @ V_basis     (r, in)  = (r, s)  @ (s, in)

    where s = small_dim << out, in. The bases are learned per weight type
    and shared across all layers. Only the coefficients are generated.
    """

    def __init__(self, out_features: int, in_features: int,
                 rank: int, small_dim: int = 32,
                 hidden_dim: int = 512):
        super().__init__()
        self.out_f = out_features
        self.in_f = in_features
        self.rank = rank
        self.small = min(small_dim, out_features, in_features)

        # Learned bases (shared across layers, updated during fine-tuning)
        self.U_basis = nn.Parameter(torch.randn(out_features, self.small) * 0.02)
        self.V_basis = nn.Parameter(torch.randn(self.small, in_features) * 0.02)

        # Coefficient generator: takes shared hidden state → (U_coeff, V_coeff)
        n_coeffs = self.small * rank * 2  # U_coeff + V_coeff
        self.gen = nn.Linear(hidden_dim, n_coeffs)
        self._reset_bases()

    def _reset_bases(self):
        """Orthonormalize bases for stable training."""
        with torch.no_grad():
            U, _, _ = torch.linalg.svd(self.U_basis, full_matrices=False)
            self.U_basis.data = U[:, :self.small]
            V, _, _ = torch.linalg.svd(self.V_basis.T, full_matrices=False)
            self.V_basis.data = V[:, :self.small].T

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coeffs = self.gen(h)
        half = coeffs.shape[-1] // 2
        U_coeff = coeffs[:half].view(self.small, self.rank)
        V_coeff = coeffs[half:].view(self.rank, self.small)

        U_eff = self.U_basis @ U_coeff
        Vh = V_coeff @ self.V_basis
        return U_eff, Vh


class WeightLaw(nn.Module):
    """The Generative Weight Law.

    Maps a layer index → all weight matrices for that transformer block.
    The law is a single nn.Module loaded once (few MB). Weights are
    generated on-the-fly, used for one forward pass, then destroyed.

    Architecture:
        layer_embedding → shared_mlp → hidden → [output_heads]
        Each head produces (U_eff, Vh) for one weight matrix.
        W ≈ U_eff @ Vh (exact at full rank, approximate at low rank).

    Args:
        d_model: model dimension (e.g., 768 for GPT-2)
        n_layers: number of transformer layers
        rank: SVD rank for weight generation (default: 64)
        small_dim: bottleneck dimension for factorized generation (default: 32)
        hidden_dim: shared MLP hidden size (default: 512)
        weight_shapes: optional dict mapping base_name → (out, in).
            Auto-detected from model if None.
    """

    def __init__(self, d_model: int, n_layers: int,
                 rank: int = 64, small_dim: int = 32,
                 hidden_dim: int = 512,
                 weight_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
                 cognitive: bool = False):
        super().__init__()
        self.rank = rank
        self.small = small_dim
        self.cognitive = cognitive

        # Layer embeddings (the ONLY per-layer storage)
        self.layer_emb = nn.Parameter(torch.randn(n_layers, d_model) * 0.02)
        self.hidden_dim = hidden_dim

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Cognitive modulator: projects hidden state → hidden space
        # Modulates weight generation per token based on current hidden state
        if cognitive:
            self.ctx_proj = nn.Linear(d_model, hidden_dim)
            # Initialize small — cognitive modulation should be subtle
            nn.init.normal_(self.ctx_proj.weight, std=0.01)
            nn.init.zeros_(self.ctx_proj.bias)

        # Output heads — one per weight type.
        # ModuleDict forbids ".", so we sanitize names.
        self._name_map: Dict[str, str] = {}  # orig_name -> safe_key
        self._reverse_map: Dict[str, str] = {}  # safe_key -> orig_name
        self.heads = nn.ModuleDict()
        if weight_shapes is not None:
            for name, (out_f, in_f) in weight_shapes.items():
                self._add_head(name, out_f, in_f)

    def _safe_key(self, name: str) -> str:
        return name.replace(".", "_").replace(",", "_")

    def _add_head(self, name: str, out_f: int, in_f: int) -> None:
        key = self._safe_key(name)
        if key not in self.heads:
            self.heads[key] = OutputHead(out_f, in_f, self.rank, self.small, self.hidden_dim)
            self._name_map[name] = key
            self._reverse_map[key] = name

    def forward(self, layer_idx: int,
                hidden_state: Optional[torch.Tensor] = None,
                field_state: Optional["FieldState"] = None,
                mode_cache: Optional["WeightModeCache"] = None
                ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate weight matrices for one layer.

        When ``field_state`` is provided, generation is selective:
        - If the layer's field score is below threshold, cached or base
          weights are returned (no computation).
        - The field ``scale`` modulates the cognitive projection magnitude.

        When ``mode_cache`` is provided, weights are cached per mode key
        and reused when the field state is similar to a previous one.

        When ``cognitive=True`` and ``hidden_state`` is provided, weights are
        modulated by the current hidden state — each token gets slightly
        different weights.

        Returns {orig_base_name: (U_eff, Vh)}.
        """
        # Check mode cache first
        if mode_cache is not None and field_state is not None:
            cached = mode_cache.get(field_state, layer_idx)
            if cached is not None:
                return cached

        h = self.mlp(self.layer_emb[layer_idx])

        if self.cognitive and hidden_state is not None:
            ctx = self.ctx_proj(hidden_state.mean(dim=1))
            if field_state is not None:
                gate = (field_state.scale * field_state.layer_score(layer_idx)).mean()
                gate = gate.clamp(max=0.5)
                h = h + ctx[0] * gate
            else:
                h = h + ctx[0]

        result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for orig_name, key in self._name_map.items():
            U_eff, Vh = self.heads[key](h)
            # Phase XII-c: scale delta by layer score for selective generation
            if field_state is not None:
                score = field_state.layer_score(layer_idx)
                if score.item() < 1.0:
                    U_eff = U_eff * score
            result[orig_name] = (U_eff, Vh)

        # Cache result
        if mode_cache is not None and field_state is not None:
            mode_cache.put(field_state, layer_idx, result)

        return result

    def generate_all(self) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Generate weights for ALL layers at once. Returns {layer_idx: ...}."""
        h_all = self.mlp(self.layer_emb)
        result: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        for i in range(self.layer_emb.shape[0]):
            h = h_all[i]
            layer: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            for orig_name, key in self._name_map.items():
                layer[orig_name] = self.heads[key](h)
            result[i] = layer
        return result

    def add_head(self, name: str, out_f: int, in_f: int):
        """Add a new weight type head (for architecture detection)."""
        self._add_head(name, out_f, in_f)


def _detect_weight_shapes(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """Detect weight matrix shapes from model's first block."""
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    block = container[0]
    shapes: Dict[str, Tuple[int, int]] = {}
    for name, param in block.named_parameters():
        if param.dim() == 2:
            base = name[:-7] if name.endswith('.weight') and len(name) > 7 else name
            shapes[base] = (param.shape[0], param.shape[1])
    return shapes


def build_law(model: nn.Module, rank: int = 64, small_dim: int = 32,
              hidden_dim: int = 512, cognitive: bool = False,
              cognitive_field: bool = False) -> WeightLaw:
    """Build a WeightLaw from a model's architecture.

    If ``cognitive=True``, the law includes a context modulator that makes
    weight generation input-dependent (per token, weights shift based on
    the current hidden state). This is the bridge to Phase XII-c (cognitive
    field): the model generates itself differently for every token.

    If ``cognitive_field=True``, also creates a ``CognitiveField`` instance
    and attaches it to the law as ``law.field`` for selective generation
    governed by Φ/A/C state.
    """
    d = _d_model(model)
    n = _n_layers(model)
    shapes = _detect_weight_shapes(model)
    law = WeightLaw(d, n, rank, small_dim, hidden_dim, shapes, cognitive=cognitive)
    if cognitive_field:
        law.field = CognitiveField(d, n, hidden_dim)
    return law


# ─── Training ────────────────────────────────────────────────────────────

def _compute_mse_loss(law: WeightLaw, targets: Dict) -> torch.Tensor:
    """MSE between generated weights and SVD targets. Used for pre-training."""
    total_loss = 0.0
    n = 0
    device = next(law.parameters()).device
    for layer_str, layer_targets in targets.items():
        layer_idx = int(layer_str)
        generated = law(layer_idx)
        for base_name, target in layer_targets.items():
            if base_name not in generated:
                continue
            U_eff, Vh = generated[base_name]
            t_W = target.get("W", None)
            if t_W is None:
                t_W = target["U_eff"] @ target["Vh"]
                target["W"] = t_W
            loss = F.mse_loss((U_eff @ Vh).float(), t_W.to(device).float())
            total_loss += loss
            n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def _assign_from_law(block: nn.Module, weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                     device: torch.device) -> None:
    """Assign generated weights to a transformer block's parameters.

    Handles freed params (shape [0]) by direct assignment.
    """
    for name, param in block.named_parameters():
        base = name[:-7] if name.endswith('.weight') and len(name) > 7 else name
        if base in weights:
            U_eff, Vh = weights[base]
            W = (U_eff @ Vh).to(device=device, dtype=param.dtype)
            if param.data.shape == W.shape:
                param.data = W
            elif param.data.shape == W.T.shape:
                param.data = W.T
            elif param.numel() == 0:
                param.data = W.contiguous()
            else:
                param.data = W.T.contiguous()
        elif name in weights:
            pass


def _count_2d_params(model: nn.Module) -> int:
    """Count number of 2D weight parameters in the model."""
    count = 0
    for p in model.parameters():
        if p.dim() == 2:
            count += 1
    return count


def pretrain_law(law: WeightLaw, targets: Dict, steps: int = 200,
                 lr: float = 1e-3, device: torch.device = torch.device("cpu"),
                 verbose: bool = True) -> List[float]:
    """Pre-train WeightLaw to match SVD targets via MSE loss."""
    law.train()
    law.to(device)
    targets_cpu = targets

    optimizer = torch.optim.AdamW(law.parameters(), lr=lr, weight_decay=1e-5)
    losses = []

    for step in range(steps):
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            loss = _compute_mse_loss(law, targets_cpu)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(law.parameters(), 1.0)
            optimizer.step()
        losses.append(loss.item())
        if verbose and (step + 1) % 50 == 0:
            print(f"  pretrain step {step+1}/{steps}: loss={loss.item():.6f}")

    law.eval()
    return losses


def _make_diff_block_kwargs(model, x, position_ids, rotary_emb, use_cache=False):
    """Build keyword args for differentiable block forward."""
    arch = _detect_arch(model)
    kwargs: Dict[str, Any] = {}
    if arch == "gpt2":
        pass
    elif arch == "llama":
        if rotary_emb is not None and hasattr(rotary_emb, "forward"):
            pid_2d = position_ids.unsqueeze(0) if position_ids.dim() == 1 else position_ids
            cos, sin = rotary_emb(x, pid_2d)
            kwargs["position_embeddings"] = (cos, sin)
    return kwargs


def _law_model_forward(model: nn.Module, input_ids: torch.Tensor,
                       law: WeightLaw,
                       use_cache: bool = False,
                       past_key_values=None) -> torch.Tensor:
    """Differentiable model forward with law-generated weights.

    Uses ``torch.func.functional_call`` per block to preserve the graph
    so gradients flow back to the law parameters.
    """
    from torch.func import functional_call

    device = _get_device(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    n_layers = len(container)

    # Shared params
    state = model.state_dict()
    non_layer = _non_layer_pattern(arch)
    shared = {k: v for k, v in state.items() if not non_layer.search(k)}
    wte, wpe, norm_w, norm_b, lm_head_w = _extract_shared(shared, arch)

    seq_len = input_ids.shape[1]
    position_ids_1d = torch.arange(seq_len, device=device, dtype=torch.long)
    x = F.embedding(input_ids, wte.to(device))
    if wpe is not None:
        x = x + F.embedding(position_ids_1d, wpe.to(device))

    rotary_emb = _get_rotary_emb(model)

    for i in range(n_layers):
        block = container[i]
        generated = law(i, hidden_state=x)

        # Build replacement state dict: generated weights override originals
        block_state = {k: v.detach() for k, v in block.state_dict().items()}
        is_delta = _is_delta_mode(law)
        for base_name, (U_eff, Vh) in generated.items():
            if U_eff.shape[-1] != Vh.shape[-2]:
                W = Vh.T @ U_eff.T
            else:
                W = U_eff @ Vh
            W = W.to(device=device)

            # Find matching param key in block
            for param_name, param in block.named_parameters():
                if not param_name.endswith(".weight"):
                    continue
                pbase = param_name[:-7]
                if pbase == base_name and W.shape == param.data.shape:
                    if is_delta:
                        W = W + param.data  # delta mode: W_eff = W_real + U@V
                    block_state[param_name] = W
                    break
                if pbase == base_name and W.T.shape == param.data.shape:
                    if is_delta:
                        W = W + param.data.T
                    block_state[param_name] = W.T
                    break

        pkv = (past_key_values[i] if (use_cache and past_key_values is not None
                                      and i < len(past_key_values)) else None)
        kwargs = _make_diff_block_kwargs(model, x, position_ids_1d, rotary_emb,
                                          use_cache=use_cache)

        args = (x,)
        fc_kwargs: Dict[str, Any] = {"use_cache": use_cache}
        if pkv is not None:
            fc_kwargs["past_key_value"] = pkv
        fc_kwargs.update(kwargs)
        out = functional_call(block, block_state, args, fc_kwargs)

        if use_cache and isinstance(out, tuple):
            x = out[0]
        else:
            x = out[0] if isinstance(out, tuple) else out

    if norm_w is not None:
        if arch == "llama":
            x = _rms_norm(x, norm_w.to(device),
                          getattr(model.config, "rms_norm_eps", 1e-6))
        else:
            x = F.layer_norm(x, (x.shape[-1],),
                             norm_w.to(device) if norm_w is not None else None,
                             norm_b.to(device) if norm_b is not None else None)
    logits = F.linear(x, lm_head_w.to(device))
    return logits


@torch.enable_grad()
def finetune_law(law: WeightLaw, model: nn.Module,
                 tokenizer, texts: List[str],
                 steps: int = 10,
                 lr: float = 1e-5,
                 max_length: int = 128,
                 device: Optional[torch.device] = None,
                 verbose: bool = True) -> List[float]:
    """Fine-tune WeightLaw end-to-end to minimize NLL on calibration data.

    Uses ``functional_call`` so gradients flow from the NLL loss all the way
    back through the generated weights into the law parameters. Model weights
    themselves are frozen — only the law is updated.

    This makes the law task-optimal: it learns to produce weights that
    minimize the actual NLL, not just the Frobenius reconstruction error.
    """
    if device is None:
        device = _get_device(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    law.train()
    law.to(device)

    optimizer = torch.optim.AdamW(law.parameters(), lr=lr, weight_decay=1e-6)
    losses = []

    for step in range(steps):
        total_loss = 0.0
        n = 0

        for text in texts:
            enc = tokenizer(text, return_tensors='pt', truncation=True,
                            max_length=max_length)
            input_ids = enc.input_ids.to(device)

            optimizer.zero_grad()
            logits = _law_model_forward(model, input_ids, law)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(law.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        losses.append(avg_loss)
        if verbose:
            print(f"  finetune step {step+1}/{steps}: NLL={avg_loss:.4f}")

    law.eval()
    return losses


# ─── Streaming with WeightLaw ────────────────────────────────────────────

def _is_delta_mode(law: WeightLaw) -> bool:
    """Detect if law is in delta mode (rank <= 8 — generates tiny deltas)."""
    return law.rank <= 8


def _assign_weights_native(block, generated, device, delta_mode=False):
    """Assign law-generated weights to block params for native forward.

    In delta mode, adds U@V as a residual on top of the real weights.
    Returns list of (param, original_tensor) for restoration.
    """
    saved = []
    for base_name, (U_eff, Vh) in generated.items():
        W = (U_eff @ Vh).to(device=device)
        for pn, param in block.named_parameters():
            if not pn.endswith(".weight"):
                continue
            pbase = pn[:-7]
            if pbase == base_name:
                if delta_mode and param.numel() > 0 and param.data.shape == W.shape:
                    saved.append((param, param.data.clone()))
                    param.data = param.data + W
                elif delta_mode and param.numel() > 0 and param.data.shape == W.T.shape:
                    saved.append((param, param.data.clone()))
                    param.data = param.data + W.T
                elif param.data.shape == W.shape:
                    param.data = W.contiguous()
                elif param.data.shape == W.T.shape:
                    param.data = W.T.contiguous()
                elif param.numel() == 0:
                    param.data = W.contiguous()
                else:
                    param.data = W.contiguous()
                break
    return saved


def _restore_weights(saved):
    """Restore weights saved by _assign_weights_native (for delta mode)."""
    for param, orig in saved:
        param.data = orig


def _make_factored_fwd(orig_fwd, U, V):
    """Create a factored forward function for one Linear layer.

    Computes y = W_real @ x + U @ (V @ x) + bias without materializing
    the full (out×in) delta weight matrix. The cost is O((out+in)×r×seq)
    instead of O(out×in×r + out×in×seq).
    """
    import functools

    U = U.contiguous()
    V = V.contiguous()

    @functools.wraps(orig_fwd)
    def _fn(x):
        base = orig_fwd(x)
        vh_x = torch.matmul(x, V.T)
        delta = torch.matmul(vh_x, U.T)
        return base + delta.to(dtype=x.dtype)

    return _fn


def _make_pure_fwd(U, V, bias):
    """Create a pure factored forward — NO real weights.

    Computes y = U @ (V @ x) + bias entirely from law-generated factors.
    The real weight (if any) is completely replaced.

    Cost: O((out+in)×r×seq) instead of O(out×in×seq).
    """
    import functools

    U = U.contiguous()
    V = V.contiguous()
    has_bias = bias is not None if isinstance(bias, torch.Tensor) else False

    @functools.wraps(lambda x: x)
    def _fn(x):
        vh_x = torch.matmul(x, V.T)
        result = torch.matmul(vh_x, U.T)
        if has_bias:
            result = result + bias
        return result

    return _fn


def _wrap_factored_forward(block, generated, device, delta_mode=True):
    """Wrap Linear layers with factored forward for law deltas.

    Instead of materializing W_eff = W_real + U@V and assigning to
    param.data, we monkey-patch each Linear's forward to compute:
        y = W_real @ x + U_eff @ (Vh @ x) + bias
    on-the-fly. This eliminates the O(out×in×r) reconstruction bottleneck.

    For non-delta mode (delta_mode=False), falls back to full weight
    assignment via _assign_weights_native (no wrapping needed).

    Returns list of (submodule, orig_forward) for _unwrap_factored_forward.
    """
    saved = []
    for base_name, (U_eff, Vh) in generated.items():
        U_d = U_eff.to(device=device, dtype=torch.float32)
        V_d = Vh.to(device=device, dtype=torch.float32)
        for pn, submod in block.named_modules():
            if isinstance(submod, nn.Linear) and pn.endswith(base_name):
                orig_fwd = submod.forward
                submod.forward = _make_factored_fwd(orig_fwd, U_d, V_d)
                saved.append((submod, orig_fwd))
                break
    return saved


def _wrap_pure_forward(block, generated, device):
    """Wrap Linear layers to use PURE factored forward — no real weights.

    Each Linear's forward is replaced with y = U @ (V @ x) + bias.
    The real weights are completely ignored (can be freed from RAM).

    This is the TRUE REVO: the model exists entirely as generated factors.
    """
    saved = []
    for base_name, (U_eff, Vh) in generated.items():
        U_d = U_eff.to(device=device, dtype=torch.float32)
        V_d = Vh.to(device=device, dtype=torch.float32)
        for pn, submod in block.named_modules():
            if isinstance(submod, nn.Linear) and pn.endswith(base_name):
                orig_fwd = submod.forward
                bias = submod.bias
                submod.forward = _make_pure_fwd(U_d, V_d, bias)
                saved.append((submod, orig_fwd))
                break
    return saved


def _unwrap_factored_forward(saved):
    """Restore original forward functions after factored forward."""
    for submod, orig_fwd in saved:
        submod.forward = orig_fwd


def _can_use_native(model: nn.Module, pure: bool = False) -> bool:
    """Check if model has valid params for native forward.

    When ``pure=True``, only non-2D params (biases, layernorms) need to be
    valid — 2D weight matrices are replaced by law-generated factors.
    """
    container = _get_layer_container(model)
    for block in container:
        for p in block.parameters():
            if p.numel() == 0 and p.dim() >= 2 and pure:
                continue  # 2D weights can be freed in pure mode
            if p.numel() == 0:
                return False
    return True


def _save_block_templates(model: nn.Module) -> Dict[int, Dict[str, torch.Tensor]]:
    """Save non-2D block parameters (layernorms, biases — tiny) before freeing."""
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    n_layers = len(container)
    templates: Dict[int, Dict[str, torch.Tensor]] = {}
    for i in range(n_layers):
        block = container[i]
        sd: Dict[str, torch.Tensor] = {}
        for name, param in block.named_parameters():
            if param.dim() < 2 and param.numel() > 0:
                sd[name] = param.data.clone()
        for name, buf in block.named_buffers():
            if buf.numel() > 0:
                sd[name] = buf.clone()
        templates[i] = sd
    return templates


@torch.no_grad()
def law_stream_forward(model: nn.Module, input_ids: torch.Tensor,
                        law: WeightLaw,
                        use_cache: bool = True,
                        past_key_values=None,
                        output_attentions: bool = False,
                        block_templates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
                        use_native: bool = True,
                        pure: bool = False,
                       ) -> torch.Tensor:
    """Forward pass with LAW-GENERATED weights streamed per layer.

    When ``pure=True``, ALL weights are generated by the law — no real model
    weights are used. Model weights can be freed before calling (requires
    ``block_templates`` for non-2D params). Each layer's Linear forward
    becomes: y = U_eff @ (Vh @ x) + bias.

    When ``pure=False`` (default), generated weights are combined with the
    real model weights (delta mode: W_eff = W_real + U@V via factored
    forward; non-delta mode: W = U@V replaces the real weight).

    When ``use_native=True`` (default for inference), generated weights are
    assigned directly to block parameters and the native block forward is
    called — no ``functional_call`` overhead. When block weights are freed
    (shape [0]), falls back to ``functional_call`` with block_templates.

    If ``block_templates`` is provided (pre-saved non-2D params from
    ``_save_block_templates``), the model's original weights may be freed
    before calling this function for maximum memory savings.

    Returns logits: (1, seq_len, vocab_size).
    """
    from torch.func import functional_call

    device = _get_device(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    n_layers = len(container)
    has_valid_params = _can_use_native(model) if use_native else False

    # Shared params
    state = model.state_dict(keep_vars=False)
    non_layer = _non_layer_pattern(arch)
    shared = {k: v for k, v in state.items() if not non_layer.search(k)}
    wte, wpe, norm_w, norm_b, lm_head_w = _extract_shared(shared, arch)

    seq_len = input_ids.shape[1]
    position_ids_1d = torch.arange(seq_len, device=device, dtype=torch.long)
    x = F.embedding(input_ids, wte.to(device))
    if wpe is not None:
        x = x + F.embedding(position_ids_1d, wpe.to(device))

    rotary_emb = _get_rotary_emb(model)

    # Save templates for potential functional_call fallback
    if not has_valid_params:
        if block_templates is not None:
            templates = block_templates
        else:
            templates = _save_block_templates(model)

    if use_cache:
        presents = []

    # Pre-compute position embeddings for LLaMA
    if arch == "llama" and rotary_emb is not None:
        pid_2d = position_ids_1d.unsqueeze(0) if position_ids_1d.dim() == 1 else position_ids_1d
        cos_sin = rotary_emb(x, pid_2d)
    else:
        cos_sin = None

    for i in range(n_layers):
        block = container[i]
        generated = law(i, hidden_state=x)

        delta_mode = _is_delta_mode(law)
        if has_valid_params:
            # === NATIVE PATH ===
            if pure:
                # Pure forward: no real weights, purely factored
                wrapped = _wrap_pure_forward(block, generated, device)
            elif delta_mode:
                # Delta forward: real weights + factored U@V
                wrapped = _wrap_factored_forward(block, generated, device)
            else:
                # Full reconstruction: U@V replaces real weights entirely
                for base_name, (U_eff, Vh) in generated.items():
                    W = (U_eff @ Vh).to(device=device)
                    for pn, param in block.named_parameters():
                        if not pn.endswith(".weight"):
                            continue
                        pbase = pn[:-7]
                        if pbase == base_name:
                            if param.data.shape == W.shape:
                                param.data = W.contiguous()
                            elif param.data.shape == W.T.shape:
                                param.data = W.T.contiguous()
                            else:
                                param.data = W.contiguous()
                            break

            pkv = (past_key_values[i] if (use_cache and past_key_values is not None
                                          and i < len(past_key_values)) else None)
            kwargs: Dict[str, Any] = {"use_cache": use_cache,
                                       "output_attentions": output_attentions}
            if pkv is not None:
                kwargs["past_key_value"] = pkv
            if cos_sin is not None:
                kwargs["position_embeddings"] = cos_sin

            out = block(x, **kwargs)
            if pure or delta_mode:
                _unwrap_factored_forward(wrapped)
            if use_cache and isinstance(out, tuple) and len(out) >= 2:
                x = out[0]
                presents.append(out[1])
            else:
                x = out[0] if isinstance(out, tuple) else out
        else:
            # === FUNCTIONAL_CALL PATH: for freed params ===
            block_state: Dict[str, torch.Tensor] = dict(templates[i])
            for base_name, (U_eff, Vh) in generated.items():
                W = (U_eff @ Vh).to(device=device)
                m, n = W.shape
                for pn, _ in block.named_parameters():
                    if not pn.endswith(".weight"):
                        continue
                    pbase = pn[:-7]
                    if pbase == base_name:
                        orig_shape = templates[i].get(pn, torch.empty(0)).shape
                        if orig_shape == W.shape:
                            block_state[pn] = W.contiguous()
                        elif orig_shape == W.T.shape:
                            block_state[pn] = W.T.contiguous()
                        elif orig_shape.numel() == 0:
                            block_state[pn] = W.contiguous()
                        else:
                            block_state[pn] = W.contiguous()
                        break

            pkv = (past_key_values[i] if (use_cache and past_key_values is not None
                                          and i < len(past_key_values)) else None)

            args = (x,)
            kwargs = {"use_cache": use_cache, "output_attentions": output_attentions}
            if pkv is not None:
                kwargs["past_key_value"] = pkv
            if cos_sin is not None:
                kwargs["position_embeddings"] = cos_sin

            out = functional_call(block, block_state, args, kwargs)
            if use_cache and isinstance(out, tuple) and len(out) >= 2:
                x = out[0]
                presents.append(out[1])
            else:
                x = out[0] if isinstance(out, tuple) else out

    if norm_w is not None:
        if arch == "llama":
            x = _rms_norm(x, norm_w.to(device),
                          getattr(model.config, "rms_norm_eps", 1e-6))
        else:
            x = F.layer_norm(x, (x.shape[-1],),
                             norm_w.to(device) if norm_w is not None else None,
                             norm_b.to(device) if norm_b is not None else None)
    logits = F.linear(x, lm_head_w.to(device))

    return logits


def law_compare_memory(model, tokenizer, law, texts,
                       max_length=128, free_blocks=False) -> Dict[str, Any]:
    """Compare full model vs law-streaming memory and NLL.

    If ``free_blocks=True``, the model's block weights are freed before law
    streaming (saving ~model_size MB). The law + block templates (~72 KB)
    remain in RAM. Peak RSS reflects: law + one generated layer's weights.
    """
    from revo.streaming import _eval_nll_full, _free_all_block_weights

    result: Dict[str, Any] = {}

    # --- Full model ---
    free_memory_trim()
    gc.collect()
    rss_before = measure_memory_rss()

    t0 = time.perf_counter()
    nll_full = _eval_nll_full(model, tokenizer, texts, max_length)
    t_full = time.perf_counter() - t0
    rss_full = measure_memory_rss()
    result["full"] = {
        "nll": nll_full,
        "time_s": t_full,
        "rss_mb": rss_full / 1024 / 1024,
        "rss_delta_mb": (rss_full - rss_before) / 1024 / 1024,
    }

    # --- Law streaming ---
    free_memory_trim()
    gc.collect()

    # Save block templates (non-2D params)
    templates = _save_block_templates(model)

    # Free block weights for true memory comparison
    if free_blocks:
        _free_all_block_weights(model)

    rss_peak = measure_memory_rss()

    def _track():
        nonlocal rss_peak
        rss_peak = max(rss_peak, measure_memory_rss())

    t0 = time.perf_counter()
    nll_law = _eval_nll_law(model, tokenizer, texts, law, max_length, _track,
                            _templates=templates)
    t_law = time.perf_counter() - t0

    result["law_streaming"] = {
        "nll": nll_law,
        "nll_delta": nll_law - nll_full,
        "time_s": t_law,
        "time_ratio": t_law / max(t_full, 1e-9),
        "rss_peak_mb": rss_peak / 1024 / 1024,
        "free_blocks": free_blocks,
    }

    n_law = sum(p.numel() for p in law.parameters())
    n_law_mb = sum(p.numel() * p.element_size() for p in law.parameters()) / 1024 / 1024
    result["law"] = {
        "n_params": n_law,
        "n_params_mb": n_law_mb,
        "rank": law.rank,
        "small_dim": law.small,
    }

    return result


def _eval_nll_law(model, tokenizer, texts, law, max_length,
                  _track_fn=None, _templates=None) -> float:
    """Evaluate NLL using law-generated weights."""
    device = _get_device(model)
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        if _track_fn:
            _track_fn()
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length)
        input_ids = enc.input_ids.to(device)
        logits = law_stream_forward(model, input_ids, law,
                                    block_templates=_templates)
        if _track_fn:
            _track_fn()
        if logits is None:
            continue
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        n_tok = shift_labels.numel()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


def _eval_nll_full_like(model, tokenizer, texts, max_length) -> float:
    from revo.streaming import _eval_nll_full
    return _eval_nll_full(model, tokenizer, texts, max_length)


@torch.no_grad()
def law_generate(model: nn.Module, tokenizer, prompt: str, law: WeightLaw,
                 max_new_tokens: int = 20,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 block_templates: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
                 verbose: bool = True,
                 use_native: bool = True,
                 pure: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Generate text with law-streamed weights and KV cache.

    When ``pure=True``, ALL weights are generated by the law — the model
    exists entirely as generated factors. Model weights can be freed before
    calling. Each Linear forward becomes: y = U_eff @ (Vh @ x) + bias.

    When ``pure=False`` (default), weights are combined with real model
    weights (delta mode: W_real + U@V; non-delta: W = U@V replaces).

    Returns:
        (generated_text, metadata)
    """
    from torch.func import functional_call

    device = _get_device(model)
    arch = _detect_arch(model)
    container = _get_layer_container(model)
    n_layers = len(container)
    has_valid_params = _can_use_native(model) if use_native else False

    # Tokenize
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    # Save templates for potential functional_call fallback
    if not has_valid_params:
        if block_templates is not None:
            templates = block_templates
        else:
            templates = _save_block_templates(model)

    # Shared params (extracted once, outside the generation loop)
    state = model.state_dict(keep_vars=False)
    non_layer = _non_layer_pattern(arch)
    shared = {k: v for k, v in state.items() if not non_layer.search(k)}
    wte, wpe, norm_w, norm_b, lm_head_w = _extract_shared(shared, arch)
    rotary_emb = _get_rotary_emb(model)

    generated = input_ids.clone()
    past_key_values: Optional[List[Any]] = None
    total_time = 0.0
    n_new = 0

    # Phase XII-c: Cognitive field and mode cache for selective generation
    has_field = hasattr(law, 'field') and law.field is not None
    mode_cache = WeightModeCache() if has_field else None
    field_state = None

    for t in range(max_new_tokens):
        t0 = time.perf_counter()
        cur_len = generated.shape[1]
        position_ids_1d = torch.arange(cur_len, device=device, dtype=torch.long)

        x = F.embedding(generated, wte.to(device))
        if wpe is not None:
            x = x + F.embedding(position_ids_1d, wpe.to(device))

        current_past = past_key_values or [None] * n_layers
        use_kv = past_key_values is not None

        # With KV cache, only the last token's hidden state is needed
        if use_kv:
            x = x[:, -1:]
            position_ids_1d = position_ids_1d[-1:]

        # Pre-compute position embeddings for LLaMA
        if arch == "llama" and rotary_emb is not None:
            pid_2d = position_ids_1d.unsqueeze(0) if position_ids_1d.dim() == 1 else position_ids_1d
            cos_sin = rotary_emb(x, pid_2d)
        else:
            cos_sin = None

        # Phase XII-c: compute cognitive field state
        if has_field:
            field_state = law.field(x)
            if verbose:
                print(f"  [{t+1}/{max_new_tokens}] {field_state}")

        presents: List[Any] = []
        for i in range(n_layers):
            block = container[i]

            # Phase XII-c: selective generation via field + cache
            if has_field and field_state is not None:
                generated_weights = law(i, hidden_state=x,
                                         field_state=field_state,
                                         mode_cache=mode_cache)
            else:
                generated_weights = law(i, hidden_state=x)

            weight_count = len(generated_weights)
            delta_mode = _is_delta_mode(law)
            if has_valid_params:
                # === NATIVE PATH ===
                if pure:
                    wrapped = _wrap_pure_forward(block, generated_weights, device)
                elif delta_mode:
                    wrapped = _wrap_factored_forward(block, generated_weights, device)
                else:
                    _assign_weights_native(block, generated_weights, device)

                pkv = current_past[i] if use_kv else None
                kwargs = {"use_cache": True, "output_attentions": False}
                if pkv is not None:
                    kwargs["past_key_value"] = pkv
                if cos_sin is not None:
                    kwargs["position_embeddings"] = cos_sin

                out = block(x, **kwargs)

                if pure or delta_mode:
                    _unwrap_factored_forward(wrapped)
            else:
                # === FUNCTIONAL_CALL PATH ===
                block_state: Dict[str, torch.Tensor] = dict(templates[i])
                for base_name, (U_eff, Vh) in generated_weights.items():
                    W = (U_eff @ Vh).to(device=device)
                    for pn, _ in block.named_parameters():
                        if not pn.endswith(".weight"):
                            continue
                        pbase = pn[:-7]
                        if pbase == base_name:
                            block_state[pn] = W.contiguous()
                            break

                pkv = current_past[i] if use_kv else None
                args = (x,)
                kwargs = {"use_cache": True, "output_attentions": False}
                if pkv is not None:
                    kwargs["past_key_value"] = pkv
                if cos_sin is not None:
                    kwargs["position_embeddings"] = cos_sin

                out = functional_call(block, block_state, args, kwargs)

            # Unified output handling
            if isinstance(out, tuple) and len(out) >= 2:
                x = out[0]
                presents.append(out[1])
            else:
                x = out[0] if isinstance(out, tuple) else out

        past_key_values = presents

        # Final norm + lm_head on last token
        x_last = x[:, -1]
        if norm_w is not None:
            if arch == "llama":
                x_last = _rms_norm(
                    x_last, norm_w.to(device),
                    getattr(model.config, "rms_norm_eps", 1e-6),
                )
            else:
                x_last = F.layer_norm(
                    x_last, (x_last.shape[-1],),
                    norm_w.to(device) if norm_w is not None else None,
                    norm_b.to(device) if norm_b is not None else None,
                )
        logits = F.linear(x_last, lm_head_w.to(device))

        # Sampling
        if temperature not in (0.0, 1.0):
            logits = logits / temperature
        if top_k is not None:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = float("-inf")
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        n_new += 1

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        if verbose:
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(f"  [{t+1}/{max_new_tokens}] {token_str} ({elapsed*1000:.0f}ms/tok)")

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    meta = {
        "n_new_tokens": n_new,
        "total_time_s": total_time,
        "mean_time_per_token_s": total_time / max(n_new, 1),
        "use_law": True,
    }
    return generated_text, meta


def law_save(law: WeightLaw, path: str) -> None:
    """Save WeightLaw to disk."""
    torch.save(law.state_dict(), path)


def law_load(law: WeightLaw, path: str, device: torch.device = None) -> WeightLaw:
    """Load WeightLaw state dict from disk."""
    sd = torch.load(path, map_location=device, weights_only=True)
    law.load_state_dict(sd)
    law.eval()
    return law


def law_info(law: WeightLaw) -> Dict[str, Any]:
    """Get size/complexity info about the law."""
    total = sum(p.numel() for p in law.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in law.parameters())
    return {
        "n_params": total,
        "size_mb": total_bytes / 1024 / 1024,
        "n_layers": law.layer_emb.shape[0],
        "rank": law.rank,
        "small_dim": law.small,
        "n_heads": len(law.heads),
        "head_names": list(law.heads.keys()),
    }
