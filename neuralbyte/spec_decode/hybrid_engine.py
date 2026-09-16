"""Hybrid CPU+GPU inference engine: PowerInfer-style sparse/dense split.

Combines two concurrent execution paths for 70B+ model inference on consumer
hardware (4GB VRAM, 16GB RAM, NVMe SSD):

  GPU Dense Path (DirectStorage):
    SSD -> PCIe -> GPU VRAM -> GDeflate decompress -> GPU compute
    Handles attention projections + hot FFN neurons. One PCIe crossing.
    Dense, regular memory access patterns = GPU excels.

  CPU Sparse Path (AVX-512 SIMD):
    Cold neuron weights in RAM -> AVX-512 gather -> sparse dot products
    Handles cold FFN neurons that activate per token (~10% fire rate).
    Irregular, scattered memory access = CPU scatter-gather handles natively.
    GPU would drop to 1% utilization on this pattern (warp divergence,
    bank conflicts, grid dimension mismatch).

  Sync:
    CPU sparse results -> PCIe -> GPU merge (64 KB activation vector).

Per-layer wall time = max(gpu_dense_ms, cpu_sparse_ms) + sync_ms.
The two paths run concurrently: while the GPU streams and computes dense
attention, the CPU computes sparse FFN activations on cold neurons already
resident in RAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import (
    HardwareProfile,
    compute_time_ms,
    gpu_compute_time_ms,
    pcie_transfer_time_ms,
    ssd_read_time_ms,
)
from .inference_engine import (
    ModelConfig,
    simulate_directstorage,
    simulate_ggml_zigzag,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridEngineConfig:
    """Configuration for the hybrid CPU+GPU inference engine.

    Models the PowerInfer-style neuron-level hot/cold split:
    - Hot neurons (frequently activated): preloaded into GPU VRAM permanently.
    - Cold neurons (rarely activated): kept in CPU RAM, computed by AVX-512.
    - Dense attention weights: streamed via DirectStorage per layer.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    d_model: int = 8192
    ffn_dim: int = 28672
    attention_ratio: float = 0.35
    ffn_ratio: float = 0.65
    sparsity: float = 0.90
    hot_neuron_ratio: float = 0.10
    avx512_speedup: float = 2.0
    gdeflate_ratio: float = 1.5
    vram_reserve_gb: float = 0.5
    sync_vector_bytes: int = 65536

    temporal_delta_ratio: float = 1.0
    speculative_tokens: int = 1
    speculative_acceptance: float = 0.60
    draft_overhead_ms: float = 80.0
    ml_cache_speedup: float = 1.0

    @property
    def params_per_layer(self) -> int:
        return self.model.params // self.model.layers

    @property
    def dense_per_layer_params(self) -> int:
        return int(self.params_per_layer * self.attention_ratio)

    @property
    def ffn_per_layer_params(self) -> int:
        return int(self.params_per_layer * self.ffn_ratio)

    @property
    def dense_per_layer_bytes(self) -> int:
        return int(self.dense_per_layer_params * self.model.bytes_per_param_int4)

    @property
    def sparse_per_layer_bytes(self) -> int:
        return int(self.ffn_per_layer_params * self.model.bytes_per_param_int4)

    @property
    def hot_neuron_params(self) -> int:
        return int(self.ffn_per_layer_params * self.hot_neuron_ratio)

    @property
    def hot_neuron_bytes(self) -> int:
        return int(self.hot_neuron_params * self.model.bytes_per_param_int4)

    @property
    def cold_neuron_params(self) -> int:
        return self.ffn_per_layer_params - self.hot_neuron_params

    @property
    def cold_neuron_bytes(self) -> int:
        return int(self.cold_neuron_params * self.model.bytes_per_param_int4)

    @property
    def hot_vram_gb(self) -> float:
        return (self.hot_neuron_bytes * self.model.layers) / (1024 ** 3)

    @property
    def cold_ram_gb(self) -> float:
        return (self.cold_neuron_bytes * self.model.layers) / (1024 ** 3)

    @property
    def active_cold_neurons(self) -> int:
        return int(self.cold_neuron_params * (1 - self.sparsity))

    @property
    def active_cold_flops(self) -> int:
        return 2 * self.active_cold_neurons

    @property
    def active_cold_bytes(self) -> int:
        return int(self.active_cold_neurons * self.model.bytes_per_param_int4)

    @property
    def dense_flops_per_layer(self) -> int:
        return 2 * self.dense_per_layer_params

    @property
    def hot_flops_per_layer(self) -> int:
        return 2 * self.hot_neuron_params

    @property
    def compressed_dense_per_layer_bytes(self) -> int:
        return int(self.dense_per_layer_bytes / self.gdeflate_ratio)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_hybrid_engine(
    config: HybridEngineConfig | None = None,
    hw: HardwareProfile | None = None,
) -> dict:
    """Simulate the hybrid CPU+GPU inference engine.

    Models concurrent GPU dense path (DirectStorage) and CPU sparse path
    (AVX-512) with per-layer timing breakdown.
    """
    if config is None:
        config = HybridEngineConfig()
    if hw is None:
        hw = HardwareProfile(ssd_sequential_gb_s=3.5, pcie_bandwidth_gb_s=12.0)

    # --- Feasibility: hot neurons must fit in VRAM ---
    vram_ok = config.hot_vram_gb + config.vram_reserve_gb <= hw.gpu_vram_gb

    if not vram_ok:
        return {
            "feasible": False,
            "fatal_reason": (
                f"Hot neurons ({config.hot_vram_gb:.2f} GB) + reserve "
                f"({config.vram_reserve_gb} GB) exceed VRAM ({hw.gpu_vram_gb} GB)"
            ),
            "gpu_io_ms": 0.0,
            "gpu_compute_ms": 0.0,
            "gpu_steady_ms": 0.0,
            "cpu_sparse_ram_ms": 0.0,
            "cpu_sparse_ssd_ms": 0.0,
            "sync_ms": 0.0,
            "per_layer_wall_ms": 0.0,
            "forward_pass_ms": 0.0,
            "ttft_ms": 0.0,
            "tokens_per_second": 0.0,
            "gpu_utilization": 0.0,
            "cpu_utilization": 0.0,
            "hot_vram_mb": round(config.hot_vram_gb * 1024, 1),
            "hot_vram_gb": round(config.hot_vram_gb, 3),
            "cold_ram_mb": round(config.cold_ram_gb * 1024, 1),
            "cold_ram_gb": round(config.cold_ram_gb, 3),
            "cold_layers_in_ram": 0,
            "cold_layers_overflow": config.model.layers,
            "transfer_ms": 0.0,
            "decompress_ms": 0.0,
            "bottleneck": "fatal",
            "per_layer": [],
        }

    # --- GPU Dense Path (per layer) ---

    compressed_dense_bytes = config.compressed_dense_per_layer_bytes
    raw_transfer_gb_s = min(hw.ssd_sequential_gb_s, hw.pcie_bandwidth_gb_s)

    effective_compressed = compressed_dense_bytes * config.temporal_delta_ratio
    transfer_ms = (effective_compressed / 1e9) / raw_transfer_gb_s * 1000

    effective_dense = config.dense_per_layer_bytes * config.temporal_delta_ratio
    decompress_ms = (effective_dense / 1e9) / hw.gpu_decompression_gb_s * 1000

    gpu_io_ms = max(transfer_ms, decompress_ms)

    gpu_compute_ms = gpu_compute_time_ms(
        config.dense_flops_per_layer + config.hot_flops_per_layer, hw
    )

    gpu_steady_ms = max(gpu_io_ms, gpu_compute_ms)

    # --- CPU Sparse Path (per layer) ---

    cpu_compute_ms = compute_time_ms(config.active_cold_flops, hw) / config.avx512_speedup

    # Cold neuron RAM overflow: how many layers fit in RAM
    if config.cold_neuron_bytes > 0:
        cold_layers_in_ram = min(
            int((hw.ram_capacity_gb * 1024 ** 3) / config.cold_neuron_bytes),
            config.model.layers,
        )
    else:
        cold_layers_in_ram = config.model.layers
    cold_layers_overflow = config.model.layers - cold_layers_in_ram

    effective_cold_bytes = int(config.active_cold_bytes * config.temporal_delta_ratio)
    cold_load_ms = ssd_read_time_ms(effective_cold_bytes, hw, sequential=True) / config.ml_cache_speedup

    cpu_sparse_ram_ms = cpu_compute_ms
    cpu_sparse_ssd_ms = cpu_compute_ms + cold_load_ms

    # --- Sync ---

    sync_ms = pcie_transfer_time_ms(config.sync_vector_bytes, hw)

    # --- Per-layer wall time ---

    first_layer_gpu = gpu_io_ms + gpu_compute_ms
    first_layer_cpu = cpu_sparse_ram_ms
    first_layer = max(first_layer_gpu, first_layer_cpu) + sync_ms

    per_layer = []
    for i in range(config.model.layers):
        is_first = i == 0
        in_ram = i < cold_layers_in_ram
        cpu_ms = cpu_sparse_ram_ms if in_ram else cpu_sparse_ssd_ms
        gpu_ms = first_layer_gpu if is_first else gpu_steady_ms
        wall = max(gpu_ms, cpu_ms) + sync_ms
        per_layer.append({
            "layer": i,
            "gpu_io_ms": round(gpu_io_ms, 4),
            "gpu_compute_ms": round(gpu_compute_ms, 4),
            "cpu_sparse_ms": round(cpu_ms, 4),
            "sync_ms": round(sync_ms, 4),
            "wall_ms": round(wall, 4),
            "cold_source": "ram" if in_ram else "ssd",
        })

    # Forward pass total
    n_remaining = config.model.layers - 1
    n_remaining_ram = max(0, cold_layers_in_ram - 1)
    n_remaining_ssd = n_remaining - n_remaining_ram

    layer_wall_ram = max(gpu_steady_ms, cpu_sparse_ram_ms) + sync_ms
    layer_wall_ssd = max(gpu_steady_ms, cpu_sparse_ssd_ms) + sync_ms

    forward_ms = first_layer + n_remaining_ram * layer_wall_ram + n_remaining_ssd * layer_wall_ssd

    # Average steady-state layer time
    if n_remaining > 0:
        avg_steady_ms = (n_remaining_ram * layer_wall_ram + n_remaining_ssd * layer_wall_ssd) / n_remaining
    else:
        avg_steady_ms = layer_wall_ram

    if config.speculative_tokens > 1:
        tokens_per_round = int(config.speculative_tokens * config.speculative_acceptance) + 1
        round_ms = forward_ms + config.draft_overhead_ms
        tokens_per_second = tokens_per_round / (round_ms / 1000) if round_ms > 0 else 0.0
    else:
        tokens_per_round = 1
        round_ms = forward_ms
        tokens_per_second = 1000.0 / (config.model.layers * avg_steady_ms) if avg_steady_ms > 0 else 0.0

    # Utilization
    gpu_utilization = gpu_compute_ms / gpu_steady_ms if gpu_steady_ms > 0 else 0.0
    cpu_utilization = cpu_sparse_ram_ms / layer_wall_ram if layer_wall_ram > 0 else 0.0

    # Bottleneck
    if gpu_io_ms >= gpu_compute_ms and gpu_io_ms >= cpu_sparse_ram_ms:
        bottleneck = "gpu_io"
    elif gpu_compute_ms >= gpu_io_ms and gpu_compute_ms >= cpu_sparse_ram_ms:
        bottleneck = "gpu_compute"
    else:
        bottleneck = "cpu_sparse"

    return {
        "feasible": True,
        "fatal_reason": None,
        "gpu_io_ms": round(gpu_io_ms, 4),
        "gpu_compute_ms": round(gpu_compute_ms, 4),
        "gpu_steady_ms": round(gpu_steady_ms, 4),
        "cpu_sparse_ram_ms": round(cpu_sparse_ram_ms, 4),
        "cpu_sparse_ssd_ms": round(cpu_sparse_ssd_ms, 4),
        "sync_ms": round(sync_ms, 4),
        "per_layer_wall_ms": round(avg_steady_ms, 4),
        "forward_pass_ms": round(forward_ms, 2),
        "ttft_ms": round(forward_ms, 2),
        "tokens_per_second": round(tokens_per_second, 4),
        "gpu_utilization": round(gpu_utilization, 4),
        "cpu_utilization": round(cpu_utilization, 4),
        "hot_vram_mb": round(config.hot_vram_gb * 1024, 1),
        "hot_vram_gb": round(config.hot_vram_gb, 3),
        "cold_ram_mb": round(config.cold_ram_gb * 1024, 1),
        "cold_ram_gb": round(config.cold_ram_gb, 3),
        "cold_layers_in_ram": cold_layers_in_ram,
        "cold_layers_overflow": cold_layers_overflow,
        "transfer_ms": round(transfer_ms, 4),
        "decompress_ms": round(decompress_ms, 4),
        "bottleneck": bottleneck,
        "per_layer": per_layer,
        "temporal_delta_ratio": config.temporal_delta_ratio,
        "speculative_tokens": config.speculative_tokens,
        "ml_cache_speedup": config.ml_cache_speedup,
        "tokens_per_round": tokens_per_round,
        "round_ms": round(round_ms, 2),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_hybrid_vs_tiers(
    config: HybridEngineConfig | None = None,
    hw: HardwareProfile | None = None,
) -> dict:
    """Compare the hybrid engine against pure GGML+ZigZag and DirectStorage."""
    if config is None:
        config = HybridEngineConfig()
    if hw is None:
        hw = HardwareProfile(ssd_sequential_gb_s=3.5, pcie_bandwidth_gb_s=12.0)

    hybrid = simulate_hybrid_engine(config, hw)
    zigzag = simulate_ggml_zigzag(hw=hw)
    ds = simulate_directstorage(hw=hw)

    speedup_vs_zigzag = 0.0
    if zigzag.feasible and zigzag.tokens_per_second > 0 and hybrid["feasible"]:
        speedup_vs_zigzag = round(hybrid["tokens_per_second"] / zigzag.tokens_per_second, 2)

    speedup_vs_ds = 0.0
    if ds.feasible and ds.tokens_per_second > 0 and hybrid["feasible"]:
        speedup_vs_ds = round(hybrid["tokens_per_second"] / ds.tokens_per_second, 2)

    return {
        "hybrid": hybrid,
        "ggml_zigzag": {
            "feasible": zigzag.feasible,
            "tokens_per_second": zigzag.tokens_per_second,
            "forward_pass_ms": zigzag.forward_pass_ms,
            "per_layer_wall_ms": zigzag.per_layer_wall_ms,
            "bottleneck": zigzag.bottleneck,
            "ram_needed_gb": zigzag.ram_needed_gb,
        },
        "directstorage": {
            "feasible": ds.feasible,
            "tokens_per_second": ds.tokens_per_second,
            "forward_pass_ms": ds.forward_pass_ms,
            "per_layer_wall_ms": ds.per_layer_wall_ms,
            "bottleneck": ds.bottleneck,
            "ram_needed_gb": ds.ram_needed_gb,
        },
        "speedup_vs_zigzag": speedup_vs_zigzag,
        "speedup_vs_directstorage": speedup_vs_ds,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_hybrid_engine_report(result: dict | None = None) -> str:
    """Generate a human-readable hybrid engine simulation report."""
    if result is None:
        result = simulate_hybrid_engine()

    lines = []
    lines.append("=" * 60)
    lines.append("  HYBRID ENGINE: CPU Sparse + GPU Dense")
    lines.append("=" * 60)
    lines.append("")

    # Architecture diagram
    lines.append("Architecture:")
    lines.append("  GPU Dense Path:   SSD -> PCIe -> VRAM -> GDeflate -> GPU compute")
    lines.append("                    (attention projections + hot FFN neurons)")
    lines.append("")
    lines.append("  CPU Sparse Path:  RAM -> AVX-512 gather -> sparse dot products")
    lines.append("                    (active cold neurons only, ~10% fire rate)")
    lines.append("")
    lines.append("  Sync:             CPU results -> PCIe -> GPU merge (64 KB)")
    lines.append("")
    lines.append("  Concurrency:      per_layer = max(GPU, CPU) + sync")
    lines.append("")

    if not result["feasible"]:
        lines.append(f"  FATAL: {result['fatal_reason']}")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # GPU Dense Path
    lines.append("--- GPU Dense Path (DirectStorage) ---")
    lines.append(f"  Compressed transfer: {result['transfer_ms']:.2f} ms/layer")
    lines.append(f"  GDeflate decompress: {result['decompress_ms']:.2f} ms/layer")
    lines.append(f"  GPU I/O (pipelined): {result['gpu_io_ms']:.2f} ms/layer")
    lines.append(f"  GPU compute:         {result['gpu_compute_ms']:.4f} ms/layer")
    lines.append(f"  Steady state:        {result['gpu_steady_ms']:.2f} ms/layer (ping-pong)")
    lines.append("")

    # CPU Sparse Path
    lines.append("--- CPU Sparse Path (AVX-512 SIMD) ---")
    lines.append(f"  CPU compute:         {result['cpu_sparse_ram_ms']:.4f} ms/layer (cold in RAM)")
    if result["cold_layers_overflow"] > 0:
        lines.append(f"  CPU compute + SSD:   {result['cpu_sparse_ssd_ms']:.4f} ms/layer (cold from SSD)")
        lines.append(f"  Cold in RAM:         {result['cold_layers_in_ram']}/{result['cold_layers_in_ram'] + result['cold_layers_overflow']} layers")
        lines.append(f"  Cold overflow:       {result['cold_layers_overflow']} layers (stream from SSD)")
    lines.append("")

    # Sync
    lines.append("--- Sync (CPU -> GPU merge) ---")
    lines.append(f"  Sync overhead:       {result['sync_ms']:.4f} ms/layer")
    lines.append("")

    # Per-layer wall time
    lines.append("--- Per-Layer Timing ---")
    lines.append(f"  Steady-state wall:   {result['per_layer_wall_ms']:.2f} ms/layer")
    lines.append(f"  Bottleneck:          {result['bottleneck']}")
    lines.append(f"  GPU utilization:     {result['gpu_utilization'] * 100:.2f}%")
    lines.append(f"  CPU utilization:     {result['cpu_utilization'] * 100:.4f}%")
    lines.append("")

    # Memory footprint
    lines.append("--- Memory Footprint ---")
    lines.append(f"  Hot neurons (VRAM):  {result['hot_vram_mb']:.0f} MB ({result['hot_vram_gb']:.2f} GB)")
    lines.append(f"  Cold neurons (RAM):  {result['cold_ram_mb']:.0f} MB ({result['cold_ram_gb']:.2f} GB)")
    lines.append("")

    # Throughput
    lines.append("--- Throughput ---")
    lines.append(f"  Forward pass:        {result['forward_pass_ms']:.0f} ms")
    lines.append(f"  TTFT:                {result['ttft_ms']:.0f} ms")
    lines.append(f"  Throughput:          {result['tokens_per_second']:.4f} tokens/sec")
    lines.append("")

    # Comparison vs tiers
    comparison = compare_hybrid_vs_tiers()
    zz = comparison["ggml_zigzag"]
    ds = comparison["directstorage"]

    lines.append("--- vs Pure Tier Engines ---")

    header = f"  {'Metric':<20} {'Hybrid':>12} {'ZigZag':>12} {'DirectStor':>12}"
    lines.append(header)
    sep = f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12}"
    lines.append(sep)

    h_tps = f"{result['tokens_per_second']:.4f}"
    zz_tps = f"{zz['tokens_per_second']:.4f}" if zz["feasible"] else "N/A"
    ds_tps = f"{ds['tokens_per_second']:.4f}" if ds["feasible"] else "N/A"
    lines.append(f"  {'Throughput (tok/s)':<20} {h_tps:>12} {zz_tps:>12} {ds_tps:>12}")

    h_wall = f"{result['per_layer_wall_ms']:.2f} ms"
    zz_wall = f"{zz['per_layer_wall_ms']:.2f} ms" if zz["feasible"] else "N/A"
    ds_wall = f"{ds['per_layer_wall_ms']:.2f} ms" if ds["feasible"] else "N/A"
    lines.append(f"  {'Per-layer wall':<20} {h_wall:>12} {zz_wall:>12} {ds_wall:>12}")

    h_fwd = f"{result['forward_pass_ms']:.0f} ms"
    zz_fwd = f"{zz['forward_pass_ms']:.0f} ms" if zz["feasible"] else "N/A"
    ds_fwd = f"{ds['forward_pass_ms']:.0f} ms" if ds["feasible"] else "N/A"
    lines.append(f"  {'Forward pass':<20} {h_fwd:>12} {zz_fwd:>12} {ds_fwd:>12}")

    lines.append("")
    if comparison["speedup_vs_zigzag"] > 0:
        lines.append(f"  Speedup vs ZigZag:       {comparison['speedup_vs_zigzag']:.1f}x")
    if comparison["speedup_vs_directstorage"] > 0:
        lines.append(f"  Speedup vs DirectStorage: {comparison['speedup_vs_directstorage']:.1f}x")
    lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O Optimizations
# ---------------------------------------------------------------------------

def compare_optimizations(hw: HardwareProfile | None = None) -> dict:
    """Compare progressive I/O optimizations on the hybrid engine.

    Returns five configurations showing cumulative improvement:
    base, temporal caching, speculative amortization, ML cache, all combined.
    """
    if hw is None:
        hw = HardwareProfile(ssd_sequential_gb_s=3.5, pcie_bandwidth_gb_s=12.0)

    configs = {
        "base": HybridEngineConfig(),
        "temporal": HybridEngineConfig(temporal_delta_ratio=0.03),
        "speculative": HybridEngineConfig(speculative_tokens=7),
        "ml_cache": HybridEngineConfig(ml_cache_speedup=2.6),
        "combined": HybridEngineConfig(
            temporal_delta_ratio=0.03,
            speculative_tokens=7,
            ml_cache_speedup=2.6,
        ),
    }

    results = {}
    for name, cfg in configs.items():
        results[name] = simulate_hybrid_engine(cfg, hw)

    base_tps = results["base"]["tokens_per_second"]
    for name in results:
        tps = results[name]["tokens_per_second"]
        results[name]["speedup_vs_base"] = round(tps / base_tps, 2) if base_tps > 0 else 0.0

    return results


def print_optimization_report(comparison: dict | None = None) -> str:
    """Generate a human-readable I/O optimization comparison report."""
    if comparison is None:
        comparison = compare_optimizations()

    lines = []
    lines.append("=" * 60)
    lines.append("  I/O OPTIMIZATIONS: Eliminating the SSD Bottleneck")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Optimizations applied:")
    lines.append("  1. Temporal Cache   — sliding window, load only delta (~3%)")
    lines.append("                        (Apple 'LLM in a Flash', Dec 2023)")
    lines.append("  2. Speculative (K=7) — draft 7 tokens, verify in one pass")
    lines.append("                        (Adaptive-EAGLE amortization)")
    lines.append("  3. ML Cache (2.6x)  — ML-predicted neuron pre-caching")
    lines.append("                        (FlashMoE, Jan 2026)")
    lines.append("")

    labels = [
        ("base", "Base (no opts)"),
        ("temporal", "+ Temporal Cache"),
        ("speculative", "+ Speculative K=7"),
        ("ml_cache", "+ ML Cache 2.6x"),
        ("combined", "All Combined"),
    ]

    header = f"  {'Config':<22} {'Wall/layer':>10} {'Forward':>10} {'tok/s':>10} {'Speedup':>8}"
    lines.append(header)
    sep = f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}"
    lines.append(sep)

    for key, label in labels:
        r = comparison[key]
        wall = f"{r['per_layer_wall_ms']:.2f} ms"
        fwd = f"{r['forward_pass_ms']:.0f} ms"
        tps = f"{r['tokens_per_second']:.2f}"
        spd = f"{r['speedup_vs_base']:.1f}x"
        lines.append(f"  {label:<22} {wall:>10} {fwd:>10} {tps:>10} {spd:>8}")

    lines.append("")

    # Analysis
    lines.append("--- Analysis ---")

    base = comparison["base"]
    temporal = comparison["temporal"]
    combined = comparison["combined"]

    temporal_reduction = 1.0 - (temporal["gpu_io_ms"] / base["gpu_io_ms"]) if base["gpu_io_ms"] > 0 else 0.0
    lines.append(f"  Temporal caching reduces GPU I/O by {temporal_reduction * 100:.0f}%")
    lines.append(f"    {base['gpu_io_ms']:.2f} ms -> {temporal['gpu_io_ms']:.2f} ms per layer")
    lines.append("")

    lines.append(f"  Speculative amortization (K=7, accept=60%):")
    spec = comparison["speculative"]
    lines.append(f"    {spec['tokens_per_round']} tokens per verification round")
    lines.append(f"    Round time: {spec['round_ms']:.0f} ms (forward + draft overhead)")
    lines.append("")

    lines.append(f"  Combined bottleneck shifts: {base['bottleneck']} -> {combined['bottleneck']}")
    lines.append(f"  Combined throughput: {combined['tokens_per_second']:.2f} tok/s ({combined['speedup_vs_base']:.0f}x over base)")
    lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
