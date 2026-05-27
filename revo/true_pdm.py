"""
True Pulse Density Modulation (PDM) Processing with Hardware-Accurate Emulation

Implements real PDM bitstream processing:
- Stochastic bitstream representation
- AND-based multiply (XOR-based add in unipolar)
- Bit-serial accumulation
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from revo._utils import orient_weight, set_by_name, skip_tied_weights, iter_linear_modules

class PDMBitstream:
    """
    Represents a value as a stochastic bitstream.
    
    In PDM:
    - Value x ∈ [0,1] is encoded as bitstream where density of 1s = x
    - For bipolar: x ∈ [-1,1] encoded as 1 for +1, 0 for -1
    """
    
    def __init__(self, value: torch.Tensor, n_bits: int = 256, bipolar: bool = True):
        """
        Create PDM bitstream from value.
        
        Args:
            value: Input values, shape [..., dim]
            n_bits: Number of bits in the stream
            bipolar: If True, use bipolar encoding (-1, +1); else unipolar (0, 1)
        """
        self.shape = value.shape
        self.n_bits = n_bits
        self.bipolar = bipolar
        self.dim = value.shape[-1]
        self.batch_size = value.numel() // self.dim
        
        # Generate stochastic bitstream
        # For bipolar: P(bit=1) = (value + 1) / 2
        if bipolar:
            prob_ones = (value + 1) / 2  # Map [-1,1] to [0,1]
            prob_ones = prob_ones.clamp(0, 1)
        else:
            prob_ones = value.clamp(0, 1)
        
        # Generate random bits
        # Shape: [batch, dim, n_bits]
        rand = torch.rand(self.batch_size, self.dim, n_bits, device=value.device)
        self.bits = (rand < prob_ones.unsqueeze(-1)).float()
        
        # Bipolar: convert 0->-1
        if bipolar:
            self.bits = self.bits * 2 - 1  # {0,1} -> {-1,+1}
        
    def to_value(self) -> torch.Tensor:
        """Reconstruct value from bitstream (mean of bits)."""
        # Average over bit dimension
        value = self.bits.mean(dim=-1)
        
        if self.bipolar:
            value = value  # Already in [-1,1]
        
        return value.view(self.shape)
    
    def __and__(self, other: 'PDMBitstream') -> 'PDMBitstream':
        """
        PDM multiply: bitwise AND of streams.
        
        For bipolar PDM, multiplication is XNOR (equivalent to AND in unipolar).
        """
        assert self.shape == other.shape
        assert self.n_bits == other.n_bits
        
        # Bitwise AND (for bipolar, this approximates multiplication)
        result_bits = self.bits * other.bits
        
        # Create new bitstream
        result = PDMBitstream.__new__(PDMBitstream)
        result.shape = self.shape
        result.n_bits = self.n_bits
        result.bipolar = self.bipolar
        result.dim = self.dim
        result.batch_size = self.batch_size
        result.bits = result_bits
        
        return result
    
    def __xor__(self, other: 'PDMBitstream') -> 'PDMBitstream':
        """
        PDM addition (saturated): bitwise XOR.
        
        In PDM, addition is approximated by XOR (saturating add).
        """
        assert self.shape == other.shape
        
        result_bits = (self.bits + other.bits).clamp(-1 if self.bipolar else 0, 1)
        
        result = PDMBitstream.__new__(PDMBitstream)
        result.shape = self.shape
        result.n_bits = self.n_bits
        result.bipolar = self.bipolar
        result.dim = self.dim
        result.batch_size = self.batch_size
        result.bits = result_bits
        
        return result


class PDMLinearLayer(nn.Module):
    """
    Linear layer using PDM bitstream processing.
    
    Implements: y = Wx + b using:
    - Bit-serial multiplication (AND)
    - Stochastic accumulation
    - Temporal integration
    """
    
    def __init__(self, in_features: int, out_features: int, bits: int = 256):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Weights in PDM format (stored as probabilities, not bitstreams)
        # Weight values in [-1, 1] -> PDM prob
        self.weight_probs = nn.Parameter(torch.rand(out_features, in_features) * 2 - 1)
        self.bias_probs = nn.Parameter(torch.rand(out_features) * 2 - 1)
        
        # For accurate emulation, we also store exact weights
        self.register_buffer('exact_weight', torch.randn(out_features, in_features))
        self.register_buffer('exact_bias', torch.randn(out_features))
        
    def pdm_matmul(self, x_bits: torch.Tensor, w_bits: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiply using PDM bitstreams.
        
        Args:
            x_bits: [batch, in, bits]
            w_bits: [out, in, bits]
        
        Returns:
            y_bits: [batch, out, bits]
        """
        batch_size = x_bits.shape[0]
        
        # For each output dimension
        results = []
        for j in range(self.out_features):
            # Compute y[j] = Σ_i x[i] * W[j,i]
            # In PDM: AND then accumulation
            
            accum = torch.zeros(batch_size, self.bits, device=x_bits.device)
            
            for i in range(self.in_features):
                # AND multiply
                prod = x_bits[:, i, :] * w_bits[j, i, :].unsqueeze(0)  # [batch, bits]
                # Accumulate (saturating)
                accum = (accum + prod).clamp(-1, 1)
            
            results.append(accum)
        
        # Stack: [batch, out, bits]
        return torch.stack(results, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using PDM processing.
        
        For efficiency, we use a hybrid approach:
        - Convert to PDM
        - Bit-serial MAC
        - Convert back
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        batch_size = x_flat.shape[0]
        
        # Convert inputs to PDM bitstreams
        x_pdm = PDMBitstream(x_flat, n_bits=self.bits, bipolar=True)
        
        # Convert weights to PDM (on the fly, since they're fixed per layer)
        w_probs = (self.weight_probs + 1) / 2  # [-1,1] -> [0,1]
        rand_w = torch.rand(self.out_features, self.in_features, self.bits, device=x.device)
        w_bits = (rand_w < w_probs.unsqueeze(-1)).float() * 2 - 1
        
        # PDM matrix multiply
        y_bits = self.pdm_matmul(x_pdm.bits, w_bits)
        
        # Reconstruct: average over bits
        y_reconstructed = y_bits.mean(dim=-1)
        
        # Add bias
        b_probs = (self.bias_probs + 1) / 2
        rand_b = torch.rand(self.out_features, self.bits, device=x.device)
        b_bits = (rand_b < b_probs.unsqueeze(-1)).float() * 2 - 1
        b_reconstructed = b_bits.mean(dim=-1).unsqueeze(0)
        
        y = y_reconstructed + b_reconstructed
        
        return y.view(*orig_shape[:-1], self.out_features)
    
    def get_pdm_accuracy(self) -> float:
        """
        Measure accuracy of PDM approximation vs exact computation.
        """
        # Sample random input
        x = torch.randn(1, self.in_features)
        
        # PDM result
        y_pdm = self.forward(x)
        
        # Exact result
        y_exact = x @ self.exact_weight.t() + self.exact_bias
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(y_pdm, y_exact, dim=-1).mean().item()
        
        return float(cos_sim)


class StochasticMultiplier(nn.Module):
    """
    Pure stochastic multiplier using bitstream operations.
    
    This is a building block for hardware PDM emulation.
    """
    
    def __init__(self, precision_bits: int = 256):
        super().__init__()
        self.precision = precision_bits
        
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Multiply two tensors using stochastic bitstreams.
        """
        # Convert both to PDM
        a_pdm = PDMBitstream(a, n_bits=self.precision, bipolar=True)
        b_pdm = PDMBitstream(b, n_bits=self.precision, bipolar=True)
        
        # AND multiply
        result_pdm = a_pdm.__and__(b_pdm)
        
        # Return value
        return result_pdm.to_value()


def replace_with_pdm(
    model: nn.Module,
    bits: int = 256,
    name_patterns: Optional[List[str]] = None,
    skip_lm_head: bool = True,
) -> Dict[str, any]:
    """
    Replace linear layers with PDM implementations.
    """
    patterns = name_patterns or ["mlp", "c_fc", "c_proj"]
    
    modules_replaced = 0
    pdm_layers = []
    
    for name, module in list(model.named_modules()):
        if skip_lm_head and name == "lm_head":
            continue
        if not any(p in name for p in patterns):
            continue
        
        if not isinstance(module, nn.Linear):
            continue
        
        # Create PDM layer
        new_layer = PDMLinearLayer(
            module.in_features,
            module.out_features,
            bits=bits
        )
        
        # Copy weights
        with torch.no_grad():
            new_layer.exact_weight.copy_(module.weight.data)
            if module.bias is not None:
                new_layer.exact_bias.copy_(module.bias.data)
            # Initialize PDM probabilities from weights
            probs = (module.weight.data + 1) / 2
            probs = probs.clamp(0, 1)
            new_layer.weight_probs.data.copy_(probs * 2 - 1)
        
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
        pdm_layers.append(new_layer)
    
    return {
        "modules_replaced": modules_replaced,
        "pdm_layers": pdm_layers,
        "bits": bits,
    }


def measure_pdm_accuracy(model: nn.Module, test_inputs: torch.Tensor) -> Dict[str, float]:
    """
    Measure overall accuracy of PDM approximation in model.
    """
    accuracies = []
    
    for module in model.modules():
        if isinstance(module, PDMLinearLayer):
            acc = module.get_pdm_accuracy()
            accuracies.append(acc)
    
    return {
        "mean_cosine_sim": sum(accuracies) / max(1, len(accuracies)),
        "min_cosine_sim": min(accuracies) if accuracies else 0.0,
        "n_layers": len(accuracies),
    }
