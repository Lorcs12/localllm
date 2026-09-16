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
from .kv_cache import KVCache, KVCacheConfig, estimate_kv_cache_sizes
from .metrics import AcceptanceTracker
from .neuron_map import NeuronMap, NeuronMapConfig, SlabIndex
from .pipeline import PipelineConfig, PipelineSimResult, print_pipeline_report, simulate_pipeline
from .predictor import CUPConfig, CUPredictor, PredictionResult, evaluate_prediction
from .slab_fetch import FetchResult, PipelineTimer, SlabFetcher
from .sparse_ops import SparseLayerConfig, estimate_full_model_sparse_time, verify_fused_correctness

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
    "PipelineConfig",
    "PipelineSimResult",
    "simulate_pipeline",
    "print_pipeline_report",
]
