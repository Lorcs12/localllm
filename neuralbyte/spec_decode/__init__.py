"""SMW-based Online Speculative Decoding (Adaptive-EAGLE).

A self-adapting draft head for speculative decoding that replaces gradient-based
online adaptation with trace-bounded Sherman-Morrison-Woodbury rank-1 updates.
Target-model rejections become free training signals absorbed in O(D^2), with
sparse margin updates touching only 2 rows of W per rejection.

The full pipeline combines five subsystems in a latency-hiding loop:
  1. CUP Predictor — maps token context to cold neuron activation probabilities
  2. Slab Fetcher — async sequential SSD reads via O_DIRECT (bypasses L3 cache)
  3. SMW Draft Head — trace-bounded adaptation from frontier model rejections
  4. Fused Gather-GEMM — sparse verification without PyTorch copy tax
  5. KV Cache — 4-bit quantization + heavy-hitter eviction for long contexts
"""

from .config import OSDConfig
from .draft_head import SMWDraftHead
from .engine import GenerationResult, SpeculativeEngine, VerificationResult
from .features import extract_draft_features, get_lm_head_weights, last_token_features
from .hardware import HardwareProfile, L3Cache, simulate_draft_window
from .inference_engine import (
    DirectStorageConfig,
    InferenceTier,
    ModelConfig,
    PyTorchConfig,
    TierResult,
    ZigZagConfig,
    compare_tiers,
    print_inference_engine_report,
    simulate_directstorage,
    simulate_ggml_zigzag,
    simulate_pytorch,
)
from .hybrid_engine import (
    HybridEngineConfig,
    compare_hybrid_vs_tiers,
    compare_optimizations,
    print_hybrid_engine_report,
    print_optimization_report,
    simulate_hybrid_engine,
)
from .hybrid_mamba import (
    ArchitectureComparison,
    HybridMambaConfig,
    ModelSpec,
    TransformerConfig,
    compare_architectures,
    print_architecture_report,
    simulate_generation_loop,
    simulate_hybrid_mamba,
    simulate_transformer,
)
from .kv_cache import KVCache, KVCacheConfig, estimate_kv_cache_sizes
from .kv_router import (
    KVPage,
    KVRouter,
    KVRouterConfig,
    print_kv_router_report,
    simulate_kv_routing,
)
from .layer_stream import (
    LayerBuffer,
    LayerStreamConfig,
    LayerStreamResult,
    LayerStreamRunner,
    StreamMode,
    compare_modes,
    print_layer_stream_report,
    simulate_layer_stream,
)
from .metrics import AcceptanceTracker
from .neuron_map import NeuronMap, NeuronMapConfig, SlabIndex
from .pipeline import PipelineConfig, PipelineSimResult, print_pipeline_report, simulate_pipeline
from .predictor import CUPConfig, CUPredictor, PredictionResult, evaluate_prediction
from .slab_fetch import FetchResult, PipelineTimer, SlabFetcher
from .sparse_ops import SparseLayerConfig, estimate_full_model_sparse_time, verify_fused_correctness
from .streaming_model import StreamingConfig, StreamingGPT2, StreamingModelAdapter, TemporalCache
from .weight_store import WeightStore

__all__ = [
    "OSDConfig",
    "SMWDraftHead",
    "SpeculativeEngine",
    "GenerationResult",
    "VerificationResult",
    "AcceptanceTracker",
    "last_token_features",
    "extract_draft_features",
    "get_lm_head_weights",
    "HardwareProfile",
    "L3Cache",
    "simulate_draft_window",
    "NeuronMap",
    "NeuronMapConfig",
    "SlabIndex",
    "CUPConfig",
    "CUPredictor",
    "PredictionResult",
    "evaluate_prediction",
    "SlabFetcher",
    "FetchResult",
    "PipelineTimer",
    "SparseLayerConfig",
    "estimate_full_model_sparse_time",
    "verify_fused_correctness",
    "KVCache",
    "KVCacheConfig",
    "estimate_kv_cache_sizes",
    "ModelSpec",
    "TransformerConfig",
    "HybridMambaConfig",
    "ArchitectureComparison",
    "simulate_transformer",
    "simulate_hybrid_mamba",
    "compare_architectures",
    "print_architecture_report",
    "simulate_generation_loop",
    "KVPage",
    "KVRouter",
    "KVRouterConfig",
    "simulate_kv_routing",
    "print_kv_router_report",
    "LayerBuffer",
    "LayerStreamConfig",
    "LayerStreamResult",
    "LayerStreamRunner",
    "StreamMode",
    "compare_modes",
    "print_layer_stream_report",
    "simulate_layer_stream",
    "InferenceTier",
    "ModelConfig",
    "PyTorchConfig",
    "ZigZagConfig",
    "DirectStorageConfig",
    "TierResult",
    "simulate_pytorch",
    "simulate_ggml_zigzag",
    "simulate_directstorage",
    "compare_tiers",
    "print_inference_engine_report",
    "HybridEngineConfig",
    "simulate_hybrid_engine",
    "compare_hybrid_vs_tiers",
    "compare_optimizations",
    "print_hybrid_engine_report",
    "print_optimization_report",
    "PipelineConfig",
    "PipelineSimResult",
    "simulate_pipeline",
    "print_pipeline_report",
    "WeightStore",
    "StreamingGPT2",
    "StreamingConfig",
    "StreamingModelAdapter",
    "TemporalCache",
]
