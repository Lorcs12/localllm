"""Complete Adaptive-EAGLE pipeline: the full 5-phase latency-hiding loop.

This is the top-level orchestrator that combines all subsystems into the
concurrent generation loop described in the paper:

  Phase 1 — CUP Prediction (RAM, ~2ms):
    The predictor maps current context to cold neuron activation probabilities.
    Identifies which slabs to fetch.

  Phase 2 — Concurrent Draft + Fetch (~80ms):
    Thread A (CPU): SMW-backed Ridge draft head generates K tokens.
    Thread B (SSD→RAM): Async slab fetcher streams cold neuron weights via
    O_DIRECT, bypassing L3 cache. GIL released via C++ extension / sleep.

  Phase 3 — Sparse Frontier Verification (~20ms):
    Fused gather-GEMM kernel processes only active neurons. KV cache uses
    4-bit quantization + heavy-hitter eviction to fit in RAM bandwidth.

  Phase 4 — Trace-Bounded SMW Update (~0.22ms):
    Rejected tokens trigger the sparse margin update: boost W[target],
    penalize W[drafted]. Trace-bounded to prevent covariance blowup.

  Phase 5 — Loop Reset (instantaneous):
    Draft head is now aligned. Resume drafting with zero SGD tax.

The pipeline simulator runs all phases with realistic timing to produce
end-to-end throughput estimates and identify which phase is the bottleneck.
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .config import OSDConfig
from .hardware import HardwareProfile, L3Cache, simulate_draft_window
from .hybrid_mamba import (
    HybridMambaConfig,
    ModelSpec,
    compare_architectures,
    simulate_generation_loop,
)
from .hybrid_engine import HybridEngineConfig, simulate_hybrid_engine
from .inference_engine import (
    InferenceTier,
    compare_tiers as compare_inference_tiers,
)
from .kv_cache import KVCache, KVCacheConfig, estimate_kv_cache_sizes
from .layer_stream import (
    LayerStreamConfig,
    StreamMode,
    estimate_memory as layer_stream_estimate_memory,
    simulate_layer_stream,
)
from .metrics import AcceptanceTracker
from .neuron_map import NeuronMap, NeuronMapConfig
from .predictor import CUPConfig, CUPredictor, PredictionResult
from .slab_fetch import FetchResult, PipelineTimer, SlabFetcher
from .sparse_ops import SparseLayerConfig, estimate_full_model_sparse_time


@dataclass(frozen=True)
class PipelineConfig:
    """Full pipeline configuration combining all subsystems."""

    osd: OSDConfig = field(default_factory=OSDConfig)
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    neuron_map: NeuronMapConfig = field(default_factory=NeuronMapConfig)
    cup: CUPConfig = field(default_factory=CUPConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    sparse: SparseLayerConfig = field(default_factory=SparseLayerConfig)

    # Layer streaming
    layer_stream: LayerStreamConfig = field(default_factory=LayerStreamConfig)
    layer_stream_mode: StreamMode = StreamMode.STREAM_1BUF

    # Hybrid Mamba architecture (optional)
    hybrid_mamba: HybridMambaConfig | None = None

    # Inference engine tier (optional)
    inference_tier: InferenceTier | None = None

    # Hybrid engine (optional)
    hybrid_engine: HybridEngineConfig | None = None

    # Pipeline parameters
    draft_tokens_per_round: int = 10
    acceptance_rate: float = 0.60
    context_length: int = 16000
    total_tokens: int = 1000

    # Cache bypass
    use_cache_bypass: bool = True


@dataclass
class PipelineRoundResult:
    """Timing breakdown for one round of the pipeline."""

    predict_ms: float
    draft_ms: float
    fetch_ms: float
    verify_ms: float
    smw_update_ms: float
    total_ms: float
    tokens_accepted: int
    tokens_drafted: int
    cache_miss: bool = False


@dataclass
class PipelineSimResult:
    """Full pipeline simulation output."""

    total_tokens: int
    total_rounds: int
    total_time_ms: float
    tokens_per_second: float
    acceptance_rate: float

    phase_totals: dict[str, float]
    bottleneck: str
    rounds: list[PipelineRoundResult]

    hardware: HardwareProfile
    kv_stats: dict
    sparse_stats: dict
    cache_bypass_stats: dict
    layer_stream_stats: dict
    architecture_stats: dict | None = None
    inference_engine_stats: dict | None = None
    hybrid_engine_stats: dict | None = None


def simulate_pipeline(config: PipelineConfig | None = None) -> PipelineSimResult:
    """Run the complete Adaptive-EAGLE pipeline simulation.

    Models all five phases with realistic hardware timing. No actual models
    are loaded — this is a physics-based throughput estimator.
    """
    if config is None:
        config = PipelineConfig()

    hw = config.hardware

    # --- Pre-compute per-phase latencies ---

    # Phase 1: CUP prediction (~2ms, lightweight MLP in RAM)
    predict_ms = 2.0

    # Phase 2: Draft window — depends on cache bypass strategy
    draft_result = simulate_draft_window(
        hw,
        draft_tokens=config.draft_tokens_per_round,
        bypass_cache=config.use_cache_bypass,
    )
    draft_ms = draft_result["total_ms"]

    # Phase 2 (concurrent): Slab fetch time
    n_cold_needed = int(config.neuron_map.n_neurons * (1 - config.neuron_map.hot_ratio) * 0.10)
    slab_count = max(1, n_cold_needed // (config.neuron_map.slab_size_bytes // config.neuron_map.neuron_size_bytes))
    fetch_bytes = slab_count * config.neuron_map.slab_size_bytes
    fetch_ms = (fetch_bytes / (hw.ssd_sequential_gb_s * 1e9)) * 1000

    # The draft and fetch run concurrently — wall clock is max(draft, fetch)
    concurrent_ms = max(draft_ms, fetch_ms)

    # Phase 3: Sparse frontier verification
    sparse_result = estimate_full_model_sparse_time(
        config.sparse, hw, num_layers=config.kv_cache.num_layers, fused=True,
    )
    sparse_verify_ms = sparse_result["total_ms"]

    # Phase 3 (continued): KV cache attention read
    kv_result = estimate_kv_cache_sizes(
        config.context_length, config.kv_cache, hw,
    )
    kv_read_ms = kv_result["optimized"]["latency_ms"]

    # Phase 3 (continued): Layer streaming overhead for target model
    ls_result = simulate_layer_stream(config.layer_stream, hw, config.layer_stream_mode)
    layer_stream_verify_ms = ls_result.total_time_ms / max(1, config.total_tokens)

    verify_ms = sparse_verify_ms + kv_read_ms + layer_stream_verify_ms

    # Phase 4: SMW update (~0.22ms per rejection)
    n_rejected = int(config.draft_tokens_per_round * (1 - config.acceptance_rate))
    smw_update_ms = n_rejected * 0.22

    # Phase 5: Loop reset (instantaneous)

    # --- Simulate generation rounds ---
    tokens_generated = 0
    rounds: list[PipelineRoundResult] = []
    total_time = 0.0

    phase_totals = {
        "predict": 0.0,
        "draft_fetch": 0.0,
        "verify": 0.0,
        "smw_update": 0.0,
    }

    while tokens_generated < config.total_tokens:
        n_drafted = config.draft_tokens_per_round
        n_accepted = int(n_drafted * config.acceptance_rate) + 1  # +1 for resampled

        round_time = predict_ms + concurrent_ms + verify_ms + smw_update_ms

        round_result = PipelineRoundResult(
            predict_ms=predict_ms,
            draft_ms=concurrent_ms,
            fetch_ms=fetch_ms,
            verify_ms=verify_ms,
            smw_update_ms=smw_update_ms,
            total_ms=round_time,
            tokens_accepted=n_accepted,
            tokens_drafted=n_drafted,
        )
        rounds.append(round_result)

        phase_totals["predict"] += predict_ms
        phase_totals["draft_fetch"] += concurrent_ms
        phase_totals["verify"] += verify_ms
        phase_totals["smw_update"] += smw_update_ms

        tokens_generated += n_accepted
        total_time += round_time

    # Phase totals as rounded values
    phase_totals = {k: round(v, 2) for k, v in phase_totals.items()}

    # Identify bottleneck
    per_round_phases = {
        "predict": predict_ms,
        "draft_fetch": concurrent_ms,
        "verify": verify_ms,
        "smw_update": smw_update_ms,
    }
    bottleneck = max(per_round_phases, key=per_round_phases.get)

    tps = (tokens_generated / total_time) * 1000 if total_time > 0 else 0

    # Cache bypass comparison
    no_bypass = simulate_draft_window(hw, draft_tokens=config.draft_tokens_per_round, bypass_cache=False)
    with_bypass = draft_result

    # Layer streaming comparison
    ls_mem_static = layer_stream_estimate_memory(config.layer_stream, StreamMode.STATIC)
    ls_mem_current = layer_stream_estimate_memory(config.layer_stream, config.layer_stream_mode)

    # Architecture comparison (hybrid mamba vs standard transformer)
    arch_stats = None
    if config.hybrid_mamba is not None:
        comparison = compare_architectures(
            context_len=config.context_length,
            hw=hw,
            model_spec=config.hybrid_mamba.spec,
        )
        gen_loop = simulate_generation_loop(config.hybrid_mamba, hw)
        arch_stats = {
            "transformer_kv_gb": comparison.transformer["kv_cache_gb"],
            "transformer_ttft_s": comparison.transformer["ttft_seconds"],
            "transformer_fatal": comparison.transformer_fatal,
            "hybrid_ram_mb": comparison.hybrid["total_ram_mb"],
            "hybrid_ttft_ms": comparison.hybrid["ttft_ms"],
            "memory_reduction_x": comparison.memory_reduction_x,
            "ttft_reduction_x": comparison.ttft_reduction_x,
            "gen_loop": gen_loop,
        }

    # Hybrid engine simulation (CPU sparse + GPU dense)
    hybrid_engine_stats = (
        simulate_hybrid_engine(config.hybrid_engine, hw)
        if config.hybrid_engine is not None
        else None
    )

    return PipelineSimResult(
        total_tokens=tokens_generated,
        total_rounds=len(rounds),
        total_time_ms=round(total_time, 2),
        tokens_per_second=round(tps, 1),
        acceptance_rate=config.acceptance_rate,
        phase_totals=phase_totals,
        bottleneck=bottleneck,
        rounds=rounds,
        hardware=hw,
        kv_stats=kv_result,
        sparse_stats=estimate_full_model_sparse_time(config.sparse, hw, fused=True),
        cache_bypass_stats={
            "without_bypass_ms": no_bypass["total_ms"],
            "with_bypass_ms": with_bypass["total_ms"],
            "bypass_saves_ms": round(no_bypass["total_ms"] - with_bypass["total_ms"], 2),
            "without_bypass_ok": no_bypass["pipeline_ok"],
            "with_bypass_ok": with_bypass["pipeline_ok"],
        },
        layer_stream_stats={
            "mode": config.layer_stream_mode.value,
            "model_size_mb": config.layer_stream.total_model_mb,
            "buffer_mb": ls_mem_current["buffer_mb"],
            "static_mb": ls_mem_static["buffer_mb"],
            "memory_reduction_x": ls_mem_current["reduction_x"],
            "stream_time_ms": round(ls_result.total_time_ms, 2),
            "overlap_saved_ms": round(ls_result.overlap_saved_ms, 2),
        },
        architecture_stats=arch_stats,
        inference_engine_stats=(
            compare_inference_tiers(hw=hw) if config.inference_tier is not None else None
        ),
        hybrid_engine_stats=hybrid_engine_stats,
    )


def print_pipeline_report(result: PipelineSimResult) -> str:
    """Generate a human-readable pipeline simulation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  ADAPTIVE-EAGLE PIPELINE SIMULATION")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Hardware:")
    hw = result.hardware
    lines.append(f"  L3 Cache:       {hw.l3_cache_mb} MB")
    lines.append(f"  RAM Bandwidth:  {hw.ram_bandwidth_gb_s} GB/s")
    lines.append(f"  SSD Sequential: {hw.ssd_sequential_gb_s} GB/s")
    lines.append(f"  CPU Compute:    {hw.cpu_tflops} TFLOPS")
    lines.append("")

    lines.append("Per-Round Breakdown:")
    r0 = result.rounds[0]
    lines.append(f"  Phase 1 - CUP Predict:     {r0.predict_ms:6.2f} ms")
    lines.append(f"  Phase 2 - Draft + Fetch:   {r0.draft_ms:6.2f} ms  (concurrent)")
    lines.append(f"    (SSD fetch:              {r0.fetch_ms:6.2f} ms, hidden)")
    lines.append(f"  Phase 3 - Verify:          {r0.verify_ms:6.2f} ms")
    lines.append(f"  Phase 4 - SMW Update:      {r0.smw_update_ms:6.2f} ms")
    lines.append(f"  TOTAL per round:           {r0.total_ms:6.2f} ms")
    lines.append(f"  Bottleneck:                {result.bottleneck}")
    lines.append("")

    lines.append("Generation:")
    lines.append(f"  Tokens generated: {result.total_tokens}")
    lines.append(f"  Rounds:           {result.total_rounds}")
    lines.append(f"  Total time:       {result.total_time_ms:.0f} ms")
    lines.append(f"  THROUGHPUT:       {result.tokens_per_second:.1f} tokens/sec")
    lines.append("")

    lines.append("Cache Bypass:")
    cb = result.cache_bypass_stats
    lines.append(f"  Without O_DIRECT: {cb['without_bypass_ms']} ms  {'OK' if cb['without_bypass_ok'] else 'FAIL'}")
    lines.append(f"  With O_DIRECT:    {cb['with_bypass_ms']} ms  {'OK' if cb['with_bypass_ok'] else 'FAIL'}")
    lines.append(f"  Savings:          {cb['bypass_saves_ms']} ms per round")
    lines.append("")

    lines.append("KV Cache:")
    kv = result.kv_stats
    lines.append(f"  Standard FP16:    {kv['standard']['cache_gb']} GB  ({kv['standard']['latency_ms']} ms)")
    lines.append(f"  Optimized INT4:   {kv['optimized']['cache_gb']} GB  ({kv['optimized']['latency_ms']} ms)")
    lines.append(f"  Reduction:        {kv['reduction_x']}x")
    lines.append("")

    lines.append("Sparse FFN:")
    sp = result.sparse_stats
    lines.append(f"  Fused gather-GEMM: {sp['total_ms']} ms across {sp['num_layers']} layers")
    lines.append("")

    lines.append("Layer Streaming:")
    ls = result.layer_stream_stats
    lines.append(f"  Mode:             {ls['mode']}")
    lines.append(f"  Model size:       {ls['model_size_mb']} MB (static)")
    lines.append(f"  Buffer size:      {ls['buffer_mb']} MB ({ls['mode']})")
    lines.append(f"  RAM reduction:    {ls['memory_reduction_x']}x")
    lines.append(f"  Stream time:      {ls['stream_time_ms']} ms")
    lines.append(f"  Overlap saved:    {ls['overlap_saved_ms']} ms")
    lines.append("")

    if result.architecture_stats is not None:
        a = result.architecture_stats
        lines.append("Architecture (Hybrid Mamba vs Transformer):")
        if a["transformer_fatal"]:
            lines.append(f"  Transformer:      {a['transformer_kv_gb']} GB KV cache (FATAL: OOM)")
        else:
            lines.append(f"  Transformer:      {a['transformer_kv_gb']} GB KV cache")
        lines.append(f"  Transformer TTFT: {a['transformer_ttft_s']} s")
        lines.append(f"  Hybrid Mamba:     {a['hybrid_ram_mb']} MB total RAM")
        lines.append(f"  Hybrid TTFT:      {a['hybrid_ttft_ms']} ms")
        lines.append(f"  Memory reduction: {a['memory_reduction_x']}x")
        lines.append(f"  TTFT reduction:   {a['ttft_reduction_x']:.0f}x")
        gl = a["gen_loop"]
        if gl["fetch_hidden"]:
            lines.append(f"  Generation:       Thread B hidden ({gl['thread_b_ms']}ms < {gl['thread_a_ms']}ms)")
        else:
            lines.append(f"  Generation:       Thread B overhead {gl['effective_overhead_ms']}ms")
        lines.append("")

    if result.inference_engine_stats is not None:
        ie = result.inference_engine_stats
        tiers = ie["tiers"]
        lines.append("Inference Engine (PyTorch vs GGML+ZigZag vs DirectStorage):")
        t1 = tiers["pytorch"]
        t2 = tiers["ggml_zigzag"]
        t3 = tiers["directstorage"]
        t1_label = "FATAL" if not t1["feasible"] else f"{t1['tokens_per_second']:.4f} tok/s"
        lines.append(f"  PyTorch:          {t1_label}")
        lines.append(f"  GGML+ZigZag:      {t2['tokens_per_second']:.4f} tok/s ({t2['ram_needed_gb']} GB RAM)")
        lines.append(f"  DirectStorage:    {t3['tokens_per_second']:.4f} tok/s ({t3['ram_needed_gb']:.2f} GB RAM)")
        lines.append(f"  DS vs ZigZag:     {ie['speedup_ds_vs_zigzag']}x speedup")
        lines.append(f"  Winner:           {ie['winner']}")
        lines.append("")

    if result.hybrid_engine_stats is not None:
        he = result.hybrid_engine_stats
        lines.append("Hybrid Engine (CPU Sparse + GPU Dense):")
        if not he["feasible"]:
            lines.append(f"  FATAL: {he['fatal_reason']}")
        else:
            lines.append(f"  GPU I/O:          {he['gpu_io_ms']:.2f} ms/layer (DirectStorage)")
            lines.append(f"  CPU Sparse:       {he['cpu_sparse_ram_ms']:.4f} ms/layer (AVX-512)")
            lines.append(f"  Forward Pass:     {he['forward_pass_ms']:.0f} ms")
            lines.append(f"  Throughput:       {he['tokens_per_second']:.4f} tok/s")
            lines.append(f"  Bottleneck:       {he['bottleneck']}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
