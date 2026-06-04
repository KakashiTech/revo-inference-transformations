"""REVO Compute: backend-agnostic reference implementation."""
from .hyperlora import HyperLoraConfig, hyperlora_generate
from .engine import apply_delta, revert_delta, DeltaHandle
from .mode_cache import ModeCache
from .potentials import log_potential
from .codebook_resonant import ResonantCodebook, ResonantCodebookConfig
from .field_selector import FieldSelector, FieldSelectorConfig
from .activation_cache import ActivationCache
from .beds import BEDSHomeostat
from .holography import HoloBoundaryAdapter
from .tqft import TopologicalProtector, TQFTConfig
from .wdm import WDMLinearWrap
from .phase_bus import PhaseBusWrap
from .reversible import ReversibleUncomputeWrap, DecomputeManager
from .holomorphic import HolomorphicFourierLinearLike
from .oscillatory_gating import OscillatoryHooks
from .hyperbolic import project_to_ball, expmap0, logmap0, mobius_add, mobius_matvec
from .hora import HoRALinearAdapter
from .spectral import spectral_prune_tensor
from .fft_kernel import CirculantLinear, nearest_circulant_first_column
from .metric_field_pinn import MetricFieldPINN, train_pinn, pde_residual
from .frequency import FreqConfig
from .equilibrium_propagation import equilibrium_propagation_tune
from .mlir_kernels import (
    compile_fn, compile_module,
    fused_circulant_forward, fused_wdm_forward,
    fused_phase_shift, fused_spectral_prune,
    hyperbolic_attention,
)
from .regimes import RegimeConfig, evaluate_regimes
from .probcal import ProbCalConfig, evaluate_probcal
from .implicit import ImplicitConfig, evaluate_implicit
from .biocomp import BioCompConfig, evaluate_biocomp
from .primitiva_router import (
    PrimitiveModel, PrimitiveRouter, PrimitiveSelector,
    DensePrimitive, CirculantPrimitive, WDMPrimitive,
    HolographyPrimitive, LowRankPrimitive,
)
from .generative_law import (
    GenerativeModel, GenerativeLayer, GenerativeLaw,
    StructureDecoder, PrimitiveBank,
)
from .streaming import (
    svd_shard_model, svd_stream_forward, svd_compare_memory,
    shard_model, stream_forward, stream_generate, compare_memory,
)
from .law_streaming import (
    WeightLaw, build_law, pretrain_law, finetune_law,
    law_stream_forward, law_generate, law_compare_memory,
    law_save, law_load, law_info,
    _save_block_templates,
    CognitiveField, FieldState, WeightModeCache,
)

__all__ = [
    "HyperLoraConfig", "hyperlora_generate",
    "apply_delta", "revert_delta", "DeltaHandle",
    "ModeCache",
    "log_potential",
    "ResonantCodebook", "ResonantCodebookConfig",
    "FieldSelector", "FieldSelectorConfig",
    "ActivationCache",
    "BEDSHomeostat",
    "HoloBoundaryAdapter",
    "TopologicalProtector", "TQFTConfig",
    "WDMLinearWrap",
    "PhaseBusWrap",
    "ReversibleUncomputeWrap", "DecomputeManager",
    "HolomorphicFourierLinearLike",
    "OscillatoryHooks",
    "project_to_ball", "expmap0", "logmap0", "mobius_add", "mobius_matvec",
    "HoRALinearAdapter",
    "spectral_prune_tensor",
    "CirculantLinear", "nearest_circulant_first_column",
    "MetricFieldPINN", "train_pinn", "pde_residual",
    "FreqConfig",
    "equilibrium_propagation_tune",
    "compile_fn", "compile_module",
    "fused_circulant_forward", "fused_wdm_forward",
    "fused_phase_shift", "fused_spectral_prune",
    "hyperbolic_attention",
    "RegimeConfig", "evaluate_regimes",
    "ProbCalConfig", "evaluate_probcal",
    "ImplicitConfig", "evaluate_implicit",
    "BioCompConfig", "evaluate_biocomp",
    "PrimitiveModel", "PrimitiveRouter", "PrimitiveSelector",
    "DensePrimitive", "CirculantPrimitive", "WDMPrimitive",
    "HolographyPrimitive", "LowRankPrimitive",
    "GenerativeModel", "GenerativeLayer", "GenerativeLaw",
    "StructureDecoder", "PrimitiveBank",
    "WeightLaw", "build_law", "pretrain_law", "finetune_law",
    "law_stream_forward", "law_generate", "law_compare_memory",
    "law_save", "law_load", "law_info",
    "CognitiveField", "FieldState", "WeightModeCache",
]

__version__ = "1.0.0"
