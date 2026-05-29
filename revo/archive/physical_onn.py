"""
Physical Oscillatory Neural Network (ONN) with Coupled Oscillator Dynamics

Implements the Kuramoto model for phase synchronization:
dφ_i/dt = ω_i + (κ/N) Σ_j sin(φ_j - φ_i)

Provides true wave interference-based computation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import set_by_name, skip_tied_weights

class CoupledOscillator(nn.Module):
    """
    A single oscillator with phase and frequency.
    
    Phase evolution follows Kuramoto dynamics when coupled.
    """
    def __init__(self, natural_freq: float = 1.0, phase: float = 0.0):
        super().__init__()
        # Natural frequency (intrinsic oscillation rate)
        self.omega = nn.Parameter(torch.tensor(natural_freq))
        
        # Current phase ( evolves as dφ/dt = ω + coupling )
        self.register_buffer('phi', torch.tensor(phase))
        
        # Amplitude (for wave superposition)
        self.amplitude = nn.Parameter(torch.tensor(1.0))
        
    def step(self, dt: float, coupling: float = 0.0):
        """
        Evolve phase by one time step.
        
        dφ/dt = ω + coupling
        """
        self.phi = self.phi + (self.omega + coupling) * dt
        # Wrap to [0, 2π]
        self.phi = self.phi % (2 * math.pi)
        
    def get_wave(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get wave value at time t: A * cos(ωt + φ)
        """
        return self.amplitude * torch.cos(self.omega * t + self.phi)


class ONNLayer(nn.Module):
    """
    Oscillatory Neural Network layer using physical wave interference.
    
    Instead of matrix multiplication, uses wave superposition:
    output[j] = Σ_i input[i] * |Σ_k A_k * cos(ω_k * t + φ_ik + φ_jk)|
    
    The interference pattern encodes the computation.
    """
    
    def __init__(self, in_features: int, out_features: int, n_oscillators: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_osc = n_oscillators
        
        # Create oscillator bank
        # Each input and output dimension gets coupled oscillators
        self.oscillators: List[List[CoupledOscillator]] = []
        
        # Coupling matrix: how much each oscillator pair influences each other
        # Shape: [in_features, out_features, n_osc]
        self.coupling_in = nn.Parameter(torch.randn(in_features, n_oscillators) * 0.1)
        self.coupling_out = nn.Parameter(torch.randn(out_features, n_oscillators) * 0.1)
        
        # Phase coupling matrix for interference
        # phi_ijk = phase between input i, output j, oscillator k
        self.phase_matrix = nn.Parameter(torch.rand(in_features, out_features, n_oscillators) * 2 * math.pi)
        
        # Natural frequencies (log-spaced for different "bands")
        freqs = torch.logspace(math.log10(0.1), math.log10(10.0), n_oscillators)
        self.register_buffer('natural_freqs', freqs)
        
        # Current time in oscillation cycle
        self.register_buffer('t', torch.tensor(0.0))
        self.dt = 0.01  # Time step
        
        # Kuramoto coupling strength
        self.kappa = nn.Parameter(torch.tensor(1.0))
        
    def kuramoto_step(self):
        """
        Perform one Kuramoto synchronization step.
        
        Updates phase coupling based on phase differences.
        """
        # Flatten phase matrix for Kuramoto dynamics
        phases = self.phase_matrix.view(-1)
        N = phases.numel()
        
        # All-to-all coupling: dφ_i/dt = (κ/N) Σ_j sin(φ_j - φ_i)
        if N > 0:
            phase_diffs = phases.unsqueeze(0) - phases.unsqueeze(1)  # [N, N]
            coupling_term = (self.kappa / N) * torch.sin(phase_diffs).sum(dim=1)
            
            # Update phases
            new_phases = (phases + coupling_term * self.dt) % (2 * math.pi)
            self.phase_matrix.data = new_phases.view_as(self.phase_matrix)
        
        self.t += self.dt
        
    def compute_interference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute output via wave interference.
        
        For each output j:
        y[j] = Σ_i x[i] * interference(i, j)
        
        where interference(i,j) = |Σ_k coupling[i,k] * coupling[j,k] * cos(ω_k * t + φ_ijk)|
        """
        batch_size = x.shape[0]
        
        # Expand for broadcasting: [batch, in, 1, osc]
        c_in = self.coupling_in.unsqueeze(0).unsqueeze(2)  # [1, in, 1, osc]
        
        # [batch, 1, out, osc]
        c_out = self.coupling_out.unsqueeze(0).unsqueeze(1)  # [1, 1, out, osc]
        
        # Phase term: cos(ωt + φ)
        # [in, out, osc]
        phase_term = torch.cos(
            self.natural_freqs.view(1, 1, -1) * self.t + self.phase_matrix
        )
        
        # Interference: product of couplings times phase
        # [batch, in, out, osc]
        interference = c_in * c_out * phase_term.unsqueeze(0)
        
        # Sum over oscillators
        interference_pattern = interference.sum(dim=-1)  # [batch, in, out]
        
        # Weighted sum over inputs
        # [batch, in] @ [batch, in, out] -> need to be careful with shapes
        # Actually: for each sample, y[j] = Σ_i x[i] * interference[i,j]
        y = torch.einsum('bi,bio->bo', x, interference_pattern)
        
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using wave interference.
        
        Performs multiple Kuramoto steps to reach synchronization,
        then computes interference pattern.
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # Evolve oscillators to synchronization
        for _ in range(10):  # Multiple sync steps
            self.kuramoto_step()
        
        # Compute interference-based output
        output = self.compute_interference(x_flat)
        
        return output.view(*orig_shape[:-1], self.out_features)
    
    def get_phase_coherence(self) -> float:
        """
        Compute phase coherence R = |<e^(iφ)>| across all oscillators.
        R = 1: perfect synchronization
        R = 0: no synchronization
        """
        phases = self.phase_matrix.view(-1)
        z = torch.exp(1j * phases)
        R = torch.abs(z.mean()).item()
        return float(R)


class WaveInterferenceProcessor(nn.Module):
    """
    Process information through wave interference across multiple frequencies.
    
    Implements a "frequency decomposition" approach where different
    semantic features resonate at different frequencies.
    """
    
    def __init__(self, dim: int, n_bands: int = 4):
        super().__init__()
        self.dim = dim
        self.n_bands = n_bands
        
        # Frequency bands (Delta, Theta, Alpha, Beta, Gamma analogy)
        self.bands = nn.ModuleList([
            ONNLayer(dim, dim, n_oscillators=2**i) 
            for i in range(n_bands)
        ])
        
        # Band mixing coefficients (learnable)
        self.band_mix = nn.Parameter(torch.ones(n_bands) / n_bands)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through multiple frequency bands and mix.
        """
        # Process each band
        band_outputs = []
        for band in self.bands:
            band_outputs.append(band(x))
        
        # Stack and weight
        stacked = torch.stack(band_outputs, dim=-1)  # [..., n_bands]
        weights = F.softmax(self.band_mix, dim=0)
        
        output = (stacked * weights.view(1, 1, -1)).sum(dim=-1)
        
        return output
    
    def get_synchronization_state(self) -> Dict[str, float]:
        """Get synchronization state across all bands."""
        coherences = [band.get_phase_coherence() for band in self.bands]
        return {
            f"band_{i}_coherence": c for i, c in enumerate(coherences)
        }


def replace_with_physical_onn(
    model: nn.Module,
    n_oscillators: int = 8,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, any]:
    """
    Replace linear layers with physical ONN implementations.
    """
    patterns = name_patterns or ["attn", "mlp", "c_fc", "c_proj"]
    
    modules_replaced = 0
    onn_layers = []
    
    for name, module in list(model.named_modules()):
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in patterns):
            continue
        
        if not isinstance(module, nn.Linear):
            continue
        
        # Create ONN layer and transfer weights
        new_layer = ONNLayer(
            module.in_features, 
            module.out_features,
            n_oscillators=n_oscillators
        )
        with torch.no_grad():
            if hasattr(module, 'weight') and module.weight is not None:
                W = module.weight.data
                sigma = W.std().item() * 0.1
                new_layer.coupling_in.data.normal_(0, sigma)
                new_layer.coupling_out.data.normal_(0, sigma)
                new_layer.phase_matrix.data = torch.rand_like(new_layer.phase_matrix) * 2 * math.pi * sigma
        
        # Replace
        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, new_layer)
        else:
            setattr(model, name, new_layer)
        
        modules_replaced += 1
        onn_layers.append(new_layer)
    
    return {
        "modules_replaced": modules_replaced,
        "onn_layers": onn_layers,
        "n_oscillators": n_oscillators,
    }


def evaluate_wave_inference(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    n_passes: int = 6,
    max_length: int = 128,
) -> Dict[str, float]:
    """
    Evaluate using wave interference averaging (multiple phase passes).
    
    This is the "true" ONN inference - running multiple times with
    different phase states and averaging (interference pattern).
    """
    model.eval()
    
    all_logits = []
    
    for pass_idx in range(n_passes):
        # Reset phases for each pass
        for module in model.modules():
            if isinstance(module, ONNLayer):
                module.phase_matrix.data = torch.rand_like(module.phase_matrix) * 2 * math.pi
                module.t = torch.tensor(0.0)
        
        pass_logits = []
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                out = model(**enc)
                pass_logits.append(out.logits[:, -1, :])  # Last token
        
        all_logits.append(torch.cat(pass_logits, dim=0))
    
    # Average logits (interference averaging)
    avg_logits = torch.stack(all_logits).mean(dim=0)
    
    # Compute metrics
    # Variance across passes (lower = more stable interference)
    logit_variance = torch.stack(all_logits).var(dim=0).mean().item()
    
    return {
        "n_passes": n_passes,
        "logit_variance": logit_variance,
        "interference_stability": 1.0 / (1.0 + logit_variance),
    }
