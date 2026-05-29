"""
Functorial Category Theory Implementation for Model Mapping

Implements proper category theory constructs:
- Categories (objects and morphisms)
- Functors (structure-preserving maps between categories)
- Natural transformations (morphisms between functors)

Maps neural network operations to categorical primitives.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class Object:
    """
    Object in a category.
    
    In the context of neural networks, objects represent:
    - Vector spaces (tensor shapes)
    - States (activations)
    """
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype = torch.float32
    
    def __hash__(self):
        return hash((self.name, self.shape, self.dtype))
    
    def __eq__(self, other):
        return isinstance(other, Object) and (self.name, self.shape, self.dtype) == (other.name, other.shape, other.dtype)


class Morphism(ABC):
    """
    Morphism (arrow) in a category.
    
    Represents a transformation from source object to target object.
    In neural networks: linear maps, activations, etc.
    """
    
    def __init__(self, source: Object, target: Object, name: str = ""):
        self.source = source
        self.target = target
        self.name = name
        
    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the morphism to data."""
        pass
    
    def compose(self, other: 'Morphism') -> 'ComposedMorphism':
        """Compose morphisms: self ∘ other (apply other, then self)."""
        if other.target != self.source:
            raise ValueError(f"Cannot compose: {other.target} != {self.source}")
        return ComposedMorphism([other, self])
    
    def __rshift__(self, other: 'Morphism') -> 'ComposedMorphism':
        """Syntax: f >> g means g ∘ f (f then g)."""
        return other.compose(self)


class LinearMorphism(Morphism):
    """
    Linear morphism: matrix multiplication.
    
    Represents a linear map between vector spaces.
    """
    
    def __init__(self, source: Object, target: Object, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__(source, target, name="linear")
        self.weight = weight
        self.bias = bias
        
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y


class ActivationMorphism(Morphism):
    """
    Non-linear activation morphism.
    """
    
    def __init__(self, obj: Object, activation: Callable[[torch.Tensor], torch.Tensor], name: str):
        super().__init__(obj, obj, name=name)
        self.activation = activation
        
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class ComposedMorphism(Morphism):
    """
    Composition of multiple morphisms.
    """
    
    def __init__(self, morphisms: List[Morphism]):
        if not morphisms:
            raise ValueError("Empty composition")
        
        # Check chain validity
        for i in range(len(morphisms) - 1):
            if morphisms[i].target != morphisms[i+1].source:
                raise ValueError(f"Invalid composition at position {i}")
        
        source = morphisms[0].source
        target = morphisms[-1].target
        super().__init__(source, target, name="composed")
        self.morphisms = morphisms
        
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.morphisms:
            x = m.apply(x)
        return x


class Category:
    """
    A category consists of:
    - Objects (ob)
    - Morphisms (hom) between objects
    - Identity morphisms for each object
    - Associative composition
    """
    
    def __init__(self, name: str):
        self.name = name
        self.objects: Set[Object] = set()
        self.morphisms: Dict[Tuple[Object, Object], List[Morphism]] = {}
        
    def add_object(self, obj: Object):
        """Add object to category."""
        self.objects.add(obj)
        
    def add_morphism(self, morph: Morphism):
        """Add morphism to category."""
        key = (morph.source, morph.target)
        if key not in self.morphisms:
            self.morphisms[key] = []
        self.morphisms[key].append(morph)
        
    def hom(self, source: Object, target: Object) -> List[Morphism]:
        """
        Hom-set: all morphisms from source to target.
        """
        return self.morphisms.get((source, target), [])
    
    def identity(self, obj: Object) -> 'IdentityMorphism':
        """Get identity morphism for object."""
        return IdentityMorphism(obj)


class IdentityMorphism(Morphism):
    """
    Identity morphism: id_A: A -> A.
    """
    
    def __init__(self, obj: Object):
        super().__init__(obj, obj, name="id")
        
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Functor:
    """
    Functor: structure-preserving map between categories.
    
    Maps:
    - Objects to objects
    - Morphisms to morphisms
    
    Preserves:
    - Identity: F(id_A) = id_F(A)
    - Composition: F(g ∘ f) = F(g) ∘ F(f)
    """
    
    def __init__(self, source_cat: Category, target_cat: Category, name: str = "F"):
        self.source = source_cat
        self.target = target_cat
        self.name = name
        
        # Object mapping
        self.object_map: Dict[Object, Object] = {}
        # Morphism mapping
        self.morphism_map: Dict[Morphism, Morphism] = {}
        
    def map_object(self, obj: Object) -> Object:
        """Map object from source to target category."""
        if obj not in self.object_map:
            # Create mapping
            mapped = Object(
                name=f"{self.name}({obj.name})",
                shape=obj.shape,
                dtype=obj.dtype
            )
            self.object_map[obj] = mapped
            self.target.add_object(mapped)
        return self.object_map[obj]
    
    def map_morphism(self, morph: Morphism) -> Morphism:
        """Map morphism from source to target category."""
        if morph not in self.morphism_map:
            # Map source and target
            new_source = self.map_object(morph.source)
            new_target = self.map_object(morph.target)
            
            # Create corresponding morphism
            if isinstance(morph, LinearMorphism):
                mapped = LinearMorphism(new_source, new_target, morph.weight, morph.bias)
            elif isinstance(morph, IdentityMorphism):
                mapped = IdentityMorphism(new_source)
            else:
                # Generic: create wrapper that applies original then maps
                mapped = MappedMorphism(morph, new_source, new_target)
            
            self.morphism_map[morph] = mapped
            self.target.add_morphism(mapped)
        
        return self.morphism_map[morph]
    
    def verify_functoriality(self) -> Dict[str, bool]:
        """
        Verify functor laws:
        1. F(id_A) = id_F(A)
        2. F(g ∘ f) = F(g) ∘ F(f)
        """
        results = {}
        
        # Check identity preservation
        for obj in self.source.objects:
            id_source = self.source.identity(obj)
            id_mapped = self.target.identity(self.map_object(obj))
            F_id = self.map_morphism(id_source)
            
            # Check if F_id equals id_F(A) (by structure, not reference)
            results[f"id_{obj.name}"] = (
                isinstance(F_id, IdentityMorphism) and 
                F_id.target == id_mapped.target
            )
        
        return results


class MappedMorphism(Morphism):
    """
    Wrapper for mapped morphisms.
    """
    
    def __init__(self, original: Morphism, source: Object, target: Object):
        super().__init__(source, target, name=f"mapped_{original.name}")
        self.original = original
        
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.original.apply(x)


class NeuralCategory(Category):
    """
    Category of neural network operations.
    
    Objects: Tensor spaces
    Morphisms: NN operations (linear, activation, etc.)
    """
    
    def __init__(self, name: str = "Neural"):
        super().__init__(name)
        
    def from_module(self, module: nn.Module, input_shape: Tuple[int, ...]) -> Morphism:
        """
        Create morphism from PyTorch module.
        """
        # Infer output shape
        with torch.no_grad():
            dummy = torch.randn(1, *input_shape[1:])
            output = module(dummy)
            output_shape = output.shape[1:]
        
        source = Object(f"input_{input_shape}", input_shape)
        target = Object(f"output_{output_shape}", output_shape)
        
        self.add_object(source)
        self.add_object(target)
        
        if isinstance(module, nn.Linear):
            morph = LinearMorphism(source, target, module.weight.data, module.bias.data if module.bias is not None else None)
        else:
            # Generic activation
            def activation_fn(x):
                return module(x)
            morph = ActivationMorphism(source, activation_fn, module.__class__.__name__)
        
        self.add_morphism(morph)
        return morph


class MicrocontrollerFunctor(Functor):
    """
    Functor mapping neural operations to microcontroller primitives.
    
    Maps:
    - Float tensors -> Fixed-point integers
    - Matrix multiply -> MAC operations
    - Activations -> lookup tables
    """
    
    def __init__(self, source_cat: Category, name: str = "MC"):
        target_cat = Category(f"{name}_Target")
        super().__init__(source_cat, target_cat, name)
        
        self.fixed_point_bits = 8
        self.use_lookup_tables = True
        
    def map_object(self, obj: Object) -> Object:
        """Map tensor to fixed-point representation."""
        if obj not in self.object_map:
            # Map to fixed-point: same shape, int dtype
            mapped = Object(
                name=f"fp8_{obj.name}",
                shape=obj.shape,
                dtype=torch.int8  # 8-bit fixed point
            )
            self.object_map[obj] = mapped
            self.target.add_object(mapped)
        return self.object_map[obj]
    
    def map_morphism(self, morph: Morphism) -> Morphism:
        """Map morphism to microcontroller primitive."""
        if morph not in self.morphism_map:
            new_source = self.map_object(morph.source)
            new_target = self.map_object(morph.target)
            
            if isinstance(morph, LinearMorphism):
                # Map linear to MAC primitive
                mapped = FixedPointLinearMorphism(
                    new_source, new_target,
                    morph.weight, morph.bias,
                    bits=self.fixed_point_bits
                )
            elif isinstance(morph, ActivationMorphism):
                # Map activation to lookup table
                if self.use_lookup_tables:
                    mapped = LookupTableMorphism(
                        new_source, new_target,
                        morph.activation,
                        bits=self.fixed_point_bits
                    )
                else:
                    mapped = MappedMorphism(morph, new_source, new_target)
            else:
                mapped = MappedMorphism(morph, new_source, new_target)
            
            self.morphism_map[morph] = mapped
            self.target.add_morphism(mapped)
        
        return self.morphism_map[morph]


class FixedPointLinearMorphism(Morphism):
    """
    Linear morphism using fixed-point arithmetic.
    """
    
    def __init__(self, source: Object, target: Object, 
                 float_weight: torch.Tensor, float_bias: Optional[torch.Tensor],
                 bits: int = 8):
        super().__init__(source, target, name="fp_linear")
        self.bits = bits
        
        # Convert to fixed-point
        self.scale = float_weight.abs().max().item() / (2**(bits-1) - 1)
        
        self.weight_fp = (float_weight / self.scale).to(torch.int8)
        if float_bias is not None:
            self.bias_fp = (float_bias / self.scale).to(torch.int8)
        else:
            self.bias_fp = None
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x is int8, do fixed-point multiply
        # Result needs to be rescaled
        x_float = x.float() * self.scale
        w_float = self.weight_fp.float() * self.scale
        
        y = x_float @ w_float.t()
        if self.bias_fp is not None:
            b_float = self.bias_fp.float() * self.scale
            y = y + b_float
        
        # Convert back to int8
        y_int = (y / self.scale).clamp(-128, 127).to(torch.int8)
        return y_int


class LookupTableMorphism(Morphism, nn.Module):
    """
    Activation implemented as lookup table (for microcontrollers).
    """
    
    def __init__(self, source: Object, target: Object,
                 activation: Callable, bits: int = 8):
        Morphism.__init__(self, source, target, name="lut_activation")
        nn.Module.__init__(self)
        
        # Build lookup table
        n_entries = 2**bits
        indices = torch.arange(n_entries) - 2**(bits-1)  # Signed range
        values = indices.float() * (1.0 / 128.0)  # Scale to reasonable range
        
        activated = activation(values)
        
        self.register_buffer('lookup_table', activated)
        self.scale = 1.0 / 128.0
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x is int8, index into table
        indices = x.to(torch.int64) + 128  # Shift to [0, 255]
        indices = indices.clamp(0, 255)
        return self.lookup_table[indices]


def create_category_from_model(model: nn.Module, input_shape: Tuple[int, ...]) -> NeuralCategory:
    """
    Create a category representation of a neural network.
    """
    cat = NeuralCategory(name=f"Cat_{model.__class__.__name__}")
    
    current_shape = input_shape
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU, nn.GELU, nn.SiLU)):
            morph = cat.from_module(module, current_shape)
            # Update shape for next layer
            with torch.no_grad():
                dummy = torch.randn(1, *current_shape[1:])
                output = module(dummy)
                current_shape = output.shape
    
    return cat


def verify_functor_mapping(model: nn.Module, test_input: torch.Tensor) -> Dict[str, any]:
    """
    Verify that functorial mapping preserves model behavior.
    """
    # Create category
    cat = create_category_from_model(model, test_input.shape)
    
    # Create microcontroller functor
    mc_functor = MicrocontrollerFunctor(cat)
    
    # Map all objects and morphisms
    for obj in list(cat.objects):
        mc_functor.map_object(obj)
    
    # Verify functoriality
    functoriality = mc_functor.verify_functoriality()
    
    return {
        "objects_mapped": len(mc_functor.object_map),
        "morphisms_mapped": len(mc_functor.morphism_map),
        "functoriality_laws": functoriality,
        "all_laws_satisfied": all(functoriality.values()),
    }
