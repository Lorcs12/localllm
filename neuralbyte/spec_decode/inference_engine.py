"""Inference engine tier comparison: three strategies for running 70B+ models
on consumer hardware (4GB VRAM, 16GB RAM, NVMe SSD).

Models the physical data paths and bottlenecks of each approach:

  Tier 1 — PyTorch Baseline:
    Standard PyTorch inference. Loads full FP16 model into RAM/VRAM.
    Path: SSD -> CPU RAM -> GPU VRAM -> Compute. Two PCIe crossings.
    For a 70B model at FP16 (~130GB), this is FATAL on consumer hardware:
    the model exceeds both 4GB VRAM and 16GB RAM.

  Tier 2 — GGML + FlexGen Zig-Zag:
    Custom C++ engine with SIMD kernels. INT4 quantized weights (~33GB).
    FlexGen's zig-zag block scheduling streams layers through a bounded
    RAM buffer in a wave pattern. Path: SSD -> RAM buffer -> GPU -> Compute.
    Two PCIe crossings, but RAM usage is bounded by buffer size (2-4GB),
    not full model size.

  Tier 3 — DirectStorage:
    Windows DirectStorage API bypasses CPU RAM entirely.
    Path: SSD -> GPU VRAM -> Compute. One PCIe crossing.
    GPU hardware decompression (GDeflate) decompresses INT4 weights on-chip.
    Zero CPU RAM usage for weights. The GPU processes layers directly from
    the SSD stream.

Key result: Tier 1 is impossible. Tier 2 is feasible but I/O-bound on the
RAM->GPU hop. Tier 3 eliminates that hop entirely, roughly doubling
throughput vs Tier 2 when SSD bandwidth is the bottleneck.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .hardware import (
    HardwareProfile,
    compute_time_ms,
    gpu_compute_time_ms,
    pcie_transfer_time_ms,
    ssd_read_time_ms,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InferenceTier(Enum):
    PYTORCH = "pytorch"
    GGML_ZIGZAG = "ggml_zigzag"
    DIRECTSTORAGE = "directstorage"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Physical parameters of a large language model for inference simulation."""

    params: int = 70_000_000_000
    layers: int = 80
    bytes_per_param_fp16: int = 2
    bytes_per_param_int4: float = 0.5
    pytorch_overhead_factor: float = 1.2

    @property
    def fp16_model_bytes(self) -> int:
        return self.params * self.bytes_per_param_fp16

    @property
    def fp16_model_gb(self) -> float:
        return self.fp16_model_bytes / (1024 ** 3)

    @property
    def int4_model_bytes(self) -> int:
        return int(self.params * self.bytes_per_param_int4)

    @property
    def int4_model_gb(self) -> float:
        return self.int4_model_bytes / (1024 ** 3)

    @property
    def quantization_ratio(self) -> float:
        return self.bytes_per_param_fp16 / self.bytes_per_param_int4

    @property
    def fp16_per_layer_bytes(self) -> int:
        return self.fp16_model_bytes // self.layers

    @property
    def int4_per_layer_bytes(self) -> int:
        return self.int4_model_bytes // self.layers

    @property
    def fp16_per_layer_mb(self) -> float:
        return self.fp16_per_layer_bytes / (1024 ** 2)

    @property
    def int4_per_layer_mb(self) -> float:
        return self.int4_per_layer_bytes / (1024 ** 2)


@dataclass(frozen=True)
class PyTorchConfig:
    """Tier 1: Standard PyTorch inference configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    activation_memory_gb: float = 2.0

    @property
    def total_ram_needed_gb(self) -> float:
        return self.model.fp16_model_gb * self.model.pytorch_overhead_factor + self.activation_memory_gb

    @property
    def total_vram_needed_gb(self) -> float:
        return self.model.fp16_model_gb + self.activation_memory_gb


@dataclass(frozen=True)
class ZigZagConfig:
    """Tier 2: GGML + FlexGen zig-zag block scheduling configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    buffer_size_gb: float = 2.0
    simd_speedup: float = 1.5

    @property
    def layers_in_buffer(self) -> int:
        per_layer_gb = self.model.int4_per_layer_bytes / (1024 ** 3)
        if per_layer_gb <= 0:
            return 1
        return max(1, int(self.buffer_size_gb / per_layer_gb))

    @property
    def total_ram_gb(self) -> float:
        return self.buffer_size_gb + 0.5


@dataclass(frozen=True)
class DirectStorageConfig:
    """Tier 3: DirectStorage + GPU hardware decompression configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    vram_buffer_layers: int = 2
    gdeflate_ratio: float = 1.5
    staging_buffer_mb: float = 64.0

    @property
    def vram_buffer_bytes(self) -> int:
        return self.vram_buffer_layers * self.model.int4_per_layer_bytes

    @property
    def vram_buffer_mb(self) -> float:
        return self.vram_buffer_bytes / (1024 ** 2)

    @property
    def compressed_per_layer_bytes(self) -> int:
        return int(self.model.int4_per_layer_bytes / self.gdeflate_ratio)

    @property
    def compressed_model_bytes(self) -> int:
        return self.compressed_per_layer_bytes * self.model.layers

    @property
    def compressed_model_gb(self) -> float:
        return self.compressed_model_bytes / (1024 ** 3)

    @property
    def total_ram_gb(self) -> float:
        return self.staging_buffer_mb / 1024


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TierResult:
    """Simulation results for one inference tier."""

    tier: InferenceTier
    feasible: bool
    fatal_reason: str | None

    model_on_disk_gb: float
    ram_needed_gb: float
    vram_needed_gb: float

    load_time_ms: float
    per_layer_io_ms: float
    per_layer_compute_ms: float
    per_layer_wall_ms: float
    forward_pass_ms: float
    ttft_ms: float

    tokens_per_second: float
    pcie_crossings: int

    bottleneck: str
    gpu_utilization: float
    io_compute_overlap: float

    per_layer: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tier 1: PyTorch
# ---------------------------------------------------------------------------

def simulate_pytorch(
    config: PyTorchConfig | None = None,
    hw: HardwareProfile | None = None,
) -> TierResult:
    """Simulate Tier 1: Standard PyTorch inference.

    Path: SSD -> CPU RAM -> GPU VRAM -> Compute. Two PCIe crossings.
    """
    if config is None:
        config = PyTorchConfig()
    if hw is None:
        hw = HardwareProfile()

    model = config.model
    fits_in_vram = config.total_vram_needed_gb <= hw.gpu_vram_gb
    fits_in_ram = config.total_ram_needed_gb <= hw.ram_capacity_gb

    if not fits_in_vram and not fits_in_ram:
        return TierResult(
            tier=InferenceTier.PYTORCH,
            feasible=False,
            fatal_reason=(
                f"Model ({model.fp16_model_gb:.1f} GB FP16) exceeds both "
                f"VRAM ({hw.gpu_vram_gb:.1f} GB) and RAM ({hw.ram_capacity_gb:.1f} GB)"
            ),
            model_on_disk_gb=round(model.fp16_model_gb, 1),
            ram_needed_gb=round(config.total_ram_needed_gb, 1),
            vram_needed_gb=round(config.total_vram_needed_gb, 1),
            load_time_ms=0.0,
            per_layer_io_ms=0.0,
            per_layer_compute_ms=0.0,
            per_layer_wall_ms=0.0,
            forward_pass_ms=0.0,
            ttft_ms=0.0,
            tokens_per_second=0.0,
            pcie_crossings=2,
            bottleneck="fatal",
            gpu_utilization=0.0,
            io_compute_overlap=0.0,
        )

    # Model fits somewhere — compute timing
    ssd_load_ms = ssd_read_time_ms(model.fp16_model_bytes, hw, sequential=True)

    if fits_in_vram:
        pcie_ms = pcie_transfer_time_ms(model.fp16_model_bytes, hw)
        load_ms = ssd_load_ms + pcie_ms
        params_per_layer = model.params // model.layers
        per_layer_compute = gpu_compute_time_ms(2 * params_per_layer, hw)
        use_gpu = True
    else:
        load_ms = ssd_load_ms
        params_per_layer = model.params // model.layers
        per_layer_compute = compute_time_ms(2 * params_per_layer, hw)
        use_gpu = False

    forward_ms = model.layers * per_layer_compute
    ttft = load_ms + forward_ms
    per_token_ms = forward_ms
    tps = 1000.0 / per_token_ms if per_token_ms > 0 else 0.0

    per_layer_data = []
    for i in range(model.layers):
        per_layer_data.append({
            "layer": i,
            "compute_ms": round(per_layer_compute, 4),
            "io_ms": 0.0,
            "wall_ms": round(per_layer_compute, 4),
        })

    return TierResult(
        tier=InferenceTier.PYTORCH,
        feasible=True,
        fatal_reason=None,
        model_on_disk_gb=round(model.fp16_model_gb, 1),
        ram_needed_gb=round(config.total_ram_needed_gb, 1),
        vram_needed_gb=round(config.total_vram_needed_gb, 1),
        load_time_ms=round(load_ms, 2),
        per_layer_io_ms=0.0,
        per_layer_compute_ms=round(per_layer_compute, 4),
        per_layer_wall_ms=round(per_layer_compute, 4),
        forward_pass_ms=round(forward_ms, 2),
        ttft_ms=round(ttft, 2),
        tokens_per_second=round(tps, 4),
        pcie_crossings=2,
        bottleneck="compute",
        gpu_utilization=1.0 if use_gpu else 0.0,
        io_compute_overlap=0.0,
        per_layer=per_layer_data,
    )


# ---------------------------------------------------------------------------
# Tier 2: GGML + FlexGen Zig-Zag
# ---------------------------------------------------------------------------

def simulate_ggml_zigzag(
    config: ZigZagConfig | None = None,
    hw: HardwareProfile | None = None,
) -> TierResult:
    """Simulate Tier 2: GGML + FlexGen zig-zag block scheduling.

    Path: SSD -> RAM buffer (zig-zag) -> GPU VRAM -> Compute. Two PCIe crossings.
    The zig-zag wave streams layers through a bounded RAM buffer. While layer L
    computes on GPU, layer L+1 loads from SSD into the RAM buffer. The buffer
    size (not model size) determines RAM footprint.
    """
    if config is None:
        config = ZigZagConfig()
    if hw is None:
        hw = HardwareProfile(ssd_sequential_gb_s=3.5, pcie_bandwidth_gb_s=12.0)

    model = config.model
    vram_per_layer_gb = model.int4_per_layer_bytes / (1024 ** 3)
    fits_in_vram = vram_per_layer_gb <= hw.gpu_vram_gb
    fits_in_ram = config.total_ram_gb <= hw.ram_capacity_gb

    if not fits_in_vram or not fits_in_ram:
        return TierResult(
            tier=InferenceTier.GGML_ZIGZAG,
            feasible=False,
            fatal_reason=(
                f"Single layer ({model.int4_per_layer_mb:.0f} MB INT4) exceeds "
                f"VRAM ({hw.gpu_vram_gb:.1f} GB) or buffer exceeds RAM"
            ),
            model_on_disk_gb=round(model.int4_model_gb, 1),
            ram_needed_gb=round(config.total_ram_gb, 1),
            vram_needed_gb=round(vram_per_layer_gb, 2),
            load_time_ms=0.0,
            per_layer_io_ms=0.0,
            per_layer_compute_ms=0.0,
            per_layer_wall_ms=0.0,
            forward_pass_ms=0.0,
            ttft_ms=0.0,
            tokens_per_second=0.0,
            pcie_crossings=2,
            bottleneck="fatal",
            gpu_utilization=0.0,
            io_compute_overlap=0.0,
        )

    # Per-layer timing
    io_ssd_ms = ssd_read_time_ms(model.int4_per_layer_bytes, hw, sequential=True)
    io_pcie_ms = pcie_transfer_time_ms(model.int4_per_layer_bytes, hw)
    total_io_ms = io_ssd_ms + io_pcie_ms

    params_per_layer = model.params // model.layers
    layer_compute_ms = gpu_compute_time_ms(2 * params_per_layer, hw) / config.simd_speedup

    # Zig-zag pipeline: SSD and PCIe share the bus, so I/O is sequential.
    # Once pipeline is full, each layer's wall time = max(total_io, compute).
    first_layer_ms = total_io_ms + layer_compute_ms
    steady_layer_ms = max(total_io_ms, layer_compute_ms)

    forward_ms = first_layer_ms + (model.layers - 1) * steady_layer_ms
    ttft_ms = forward_ms
    per_token_ms = model.layers * steady_layer_ms
    tps = 1000.0 / per_token_ms if per_token_ms > 0 else 0.0

    if steady_layer_ms > 0:
        gpu_util = layer_compute_ms / steady_layer_ms
        overlap = 1.0 - (steady_layer_ms / (total_io_ms + layer_compute_ms))
    else:
        gpu_util = 0.0
        overlap = 0.0

    if total_io_ms >= layer_compute_ms:
        bottleneck = "io_ssd" if io_ssd_ms >= io_pcie_ms else "io_pcie"
    else:
        bottleneck = "compute"

    per_layer_data = []
    for i in range(model.layers):
        wall = first_layer_ms if i == 0 else steady_layer_ms
        per_layer_data.append({
            "layer": i,
            "io_ssd_ms": round(io_ssd_ms, 4),
            "io_pcie_ms": round(io_pcie_ms, 4),
            "compute_ms": round(layer_compute_ms, 4),
            "wall_ms": round(wall, 4),
        })

    return TierResult(
        tier=InferenceTier.GGML_ZIGZAG,
        feasible=True,
        fatal_reason=None,
        model_on_disk_gb=round(model.int4_model_gb, 1),
        ram_needed_gb=round(config.total_ram_gb, 1),
        vram_needed_gb=round(vram_per_layer_gb, 2),
        load_time_ms=round(first_layer_ms, 2),
        per_layer_io_ms=round(total_io_ms, 4),
        per_layer_compute_ms=round(layer_compute_ms, 4),
        per_layer_wall_ms=round(steady_layer_ms, 4),
        forward_pass_ms=round(forward_ms, 2),
        ttft_ms=round(ttft_ms, 2),
        tokens_per_second=round(tps, 4),
        pcie_crossings=2,
        bottleneck=bottleneck,
        gpu_utilization=round(gpu_util, 4),
        io_compute_overlap=round(max(0.0, overlap), 4),
        per_layer=per_layer_data,
    )


# ---------------------------------------------------------------------------
# Tier 3: DirectStorage
# ---------------------------------------------------------------------------

def simulate_directstorage(
    config: DirectStorageConfig | None = None,
    hw: HardwareProfile | None = None,
) -> TierResult:
    """Simulate Tier 3: DirectStorage + GPU hardware decompression.

    Path: SSD -> GPU VRAM -> Compute. One PCIe crossing.
    GDeflate decompresses INT4 weights on the GPU's dedicated hardware unit.
    Zero CPU RAM usage for weights.
    """
    if config is None:
        config = DirectStorageConfig()
    if hw is None:
        hw = HardwareProfile(ssd_sequential_gb_s=3.5, pcie_bandwidth_gb_s=12.0)

    model = config.model
    vram_buffer_gb = config.vram_buffer_bytes / (1024 ** 3)
    fits_in_vram = vram_buffer_gb <= hw.gpu_vram_gb

    if not fits_in_vram:
        return TierResult(
            tier=InferenceTier.DIRECTSTORAGE,
            feasible=False,
            fatal_reason=(
                f"VRAM buffer ({config.vram_buffer_mb:.0f} MB for "
                f"{config.vram_buffer_layers} layers) exceeds VRAM ({hw.gpu_vram_gb:.1f} GB)"
            ),
            model_on_disk_gb=round(config.compressed_model_gb, 1),
            ram_needed_gb=round(config.total_ram_gb, 2),
            vram_needed_gb=round(vram_buffer_gb, 2),
            load_time_ms=0.0,
            per_layer_io_ms=0.0,
            per_layer_compute_ms=0.0,
            per_layer_wall_ms=0.0,
            forward_pass_ms=0.0,
            ttft_ms=0.0,
            tokens_per_second=0.0,
            pcie_crossings=1,
            bottleneck="fatal",
            gpu_utilization=0.0,
            io_compute_overlap=0.0,
        )

    # Transfer bottleneck: SSD → PCIe → GPU (single hop)
    raw_transfer_gb_s = min(hw.ssd_sequential_gb_s, hw.pcie_bandwidth_gb_s)

    # Compressed data transfer time per layer
    compressed_bytes = config.compressed_per_layer_bytes
    transfer_ms = (compressed_bytes / 1e9) / raw_transfer_gb_s * 1000

    # GPU hardware decompression (pipelined with transfer on different HW unit)
    decompress_ms = (model.int4_per_layer_bytes / 1e9) / hw.gpu_decompression_gb_s * 1000
    io_ms = max(transfer_ms, decompress_ms)

    # GPU compute per layer
    params_per_layer = model.params // model.layers
    layer_compute_ms = gpu_compute_time_ms(2 * params_per_layer, hw)

    # Ping-pong in VRAM: while layer L computes, layer L+1 loads
    first_layer_ms = io_ms + layer_compute_ms
    steady_layer_ms = max(io_ms, layer_compute_ms)

    forward_ms = first_layer_ms + (model.layers - 1) * steady_layer_ms
    ttft_ms = forward_ms
    per_token_ms = model.layers * steady_layer_ms
    tps = 1000.0 / per_token_ms if per_token_ms > 0 else 0.0

    if steady_layer_ms > 0:
        gpu_util = layer_compute_ms / steady_layer_ms
        overlap = 1.0 - (steady_layer_ms / (io_ms + layer_compute_ms))
    else:
        gpu_util = 0.0
        overlap = 0.0

    if io_ms >= layer_compute_ms:
        ssd_bottleneck_ms = (compressed_bytes / 1e9) / hw.ssd_sequential_gb_s * 1000
        pcie_bottleneck_ms = (compressed_bytes / 1e9) / hw.pcie_bandwidth_gb_s * 1000
        if ssd_bottleneck_ms >= pcie_bottleneck_ms:
            bottleneck = "io_ssd"
        else:
            bottleneck = "io_pcie"
    else:
        bottleneck = "compute"

    per_layer_data = []
    for i in range(model.layers):
        wall = first_layer_ms if i == 0 else steady_layer_ms
        per_layer_data.append({
            "layer": i,
            "transfer_ms": round(transfer_ms, 4),
            "decompress_ms": round(decompress_ms, 4),
            "compute_ms": round(layer_compute_ms, 4),
            "wall_ms": round(wall, 4),
        })

    return TierResult(
        tier=InferenceTier.DIRECTSTORAGE,
        feasible=True,
        fatal_reason=None,
        model_on_disk_gb=round(config.compressed_model_gb, 1),
        ram_needed_gb=round(config.total_ram_gb, 2),
        vram_needed_gb=round(vram_buffer_gb, 2),
        load_time_ms=round(first_layer_ms, 2),
        per_layer_io_ms=round(io_ms, 4),
        per_layer_compute_ms=round(layer_compute_ms, 4),
        per_layer_wall_ms=round(steady_layer_ms, 4),
        forward_pass_ms=round(forward_ms, 2),
        ttft_ms=round(ttft_ms, 2),
        tokens_per_second=round(tps, 4),
        pcie_crossings=1,
        bottleneck=bottleneck,
        gpu_utilization=round(gpu_util, 4),
        io_compute_overlap=round(max(0.0, overlap), 4),
        per_layer=per_layer_data,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_tiers(
    model: ModelConfig | None = None,
    hw: HardwareProfile | None = None,
    zigzag_buffer_gb: float | None = None,
) -> dict:
    """Run all three tier simulations and return a side-by-side comparison."""
    if model is None:
        model = ModelConfig()
    if hw is None:
        hw = HardwareProfile(
            ssd_sequential_gb_s=3.5,
            pcie_bandwidth_gb_s=12.0,
            gpu_vram_gb=4.0,
            gpu_tflops=2.984,
            ram_capacity_gb=16.0,
            cpu_tflops=0.5,
            ram_bandwidth_gb_s=40.0,
        )

    pytorch_cfg = PyTorchConfig(model=model)
    zigzag_cfg = ZigZagConfig(
        model=model,
        buffer_size_gb=zigzag_buffer_gb if zigzag_buffer_gb is not None else 2.0,
    )
    ds_cfg = DirectStorageConfig(model=model)

    t1 = simulate_pytorch(pytorch_cfg, hw)
    t2 = simulate_ggml_zigzag(zigzag_cfg, hw)
    t3 = simulate_directstorage(ds_cfg, hw)

    tiers = {
        "pytorch": _tier_to_dict(t1),
        "ggml_zigzag": _tier_to_dict(t2),
        "directstorage": _tier_to_dict(t3),
    }

    # Determine winner: best feasible throughput
    feasible = [(name, t) for name, t in [("pytorch", t1), ("ggml_zigzag", t2), ("directstorage", t3)] if t.feasible]
    if feasible:
        winner = max(feasible, key=lambda x: x[1].tokens_per_second)[0]
    else:
        winner = None

    # Speedup of Tier 3 vs Tier 2
    speedup_ds_vs_zz = 0.0
    if t2.feasible and t2.tokens_per_second > 0 and t3.feasible:
        speedup_ds_vs_zz = round(t3.tokens_per_second / t2.tokens_per_second, 2)

    return {
        "model": {
            "params": model.params,
            "layers": model.layers,
            "fp16_gb": round(model.fp16_model_gb, 1),
            "int4_gb": round(model.int4_model_gb, 1),
            "quantization_ratio": model.quantization_ratio,
        },
        "hardware": {
            "gpu_vram_gb": hw.gpu_vram_gb,
            "gpu_tflops": hw.gpu_tflops,
            "ram_capacity_gb": hw.ram_capacity_gb,
            "ssd_sequential_gb_s": hw.ssd_sequential_gb_s,
            "pcie_bandwidth_gb_s": hw.pcie_bandwidth_gb_s,
        },
        "tiers": tiers,
        "winner": winner,
        "speedup_ds_vs_zigzag": speedup_ds_vs_zz,
    }


def _tier_to_dict(result: TierResult) -> dict:
    """Convert a TierResult to a plain dict for the comparison output."""
    return {
        "tier": result.tier.value,
        "feasible": result.feasible,
        "fatal_reason": result.fatal_reason,
        "model_on_disk_gb": result.model_on_disk_gb,
        "ram_needed_gb": result.ram_needed_gb,
        "vram_needed_gb": result.vram_needed_gb,
        "load_time_ms": result.load_time_ms,
        "per_layer_io_ms": result.per_layer_io_ms,
        "per_layer_compute_ms": result.per_layer_compute_ms,
        "per_layer_wall_ms": result.per_layer_wall_ms,
        "forward_pass_ms": result.forward_pass_ms,
        "ttft_ms": result.ttft_ms,
        "tokens_per_second": result.tokens_per_second,
        "pcie_crossings": result.pcie_crossings,
        "bottleneck": result.bottleneck,
        "gpu_utilization": result.gpu_utilization,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_inference_engine_report(comparison: dict | None = None) -> str:
    """Generate a human-readable inference engine tier comparison report."""
    if comparison is None:
        comparison = compare_tiers()

    m = comparison["model"]
    h = comparison["hardware"]
    tiers = comparison["tiers"]

    lines = []
    lines.append("=" * 60)
    lines.append(f"  INFERENCE ENGINE: {m['params'] // 1_000_000_000}B Model on Consumer Hardware")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Model:")
    lines.append(f"  Parameters:      {m['params']:,} ({m['params'] // 1_000_000_000}B)")
    lines.append(f"  Layers:          {m['layers']}")
    lines.append(f"  FP16 size:       {m['fp16_gb']} GB")
    lines.append(f"  INT4 size:       {m['int4_gb']} GB")
    lines.append(f"  Quantization:    {m['quantization_ratio']}x compression")
    lines.append("")

    lines.append("Hardware:")
    lines.append(f"  GPU:             {h['gpu_vram_gb']} GB VRAM, {h['gpu_tflops']} TFLOPS")
    lines.append(f"  RAM:             {h['ram_capacity_gb']} GB")
    lines.append(f"  SSD:             {h['ssd_sequential_gb_s']} GB/s sequential")
    lines.append(f"  PCIe:            {h['pcie_bandwidth_gb_s']} GB/s")
    lines.append("")

    # Tier 1: PyTorch
    t1 = tiers["pytorch"]
    lines.append("--- TIER 1: PyTorch Baseline (FP16) ---")
    lines.append(f"  Path:            SSD -> CPU RAM -> GPU VRAM -> Compute")
    lines.append(f"  Model on disk:   {t1['model_on_disk_gb']} GB")
    lines.append(f"  RAM needed:      {t1['ram_needed_gb']} GB")
    lines.append(f"  VRAM needed:     {t1['vram_needed_gb']} GB")
    if not t1["feasible"]:
        lines.append(f"  FATAL: {t1['fatal_reason']}")
        lines.append("  Result: IMPOSSIBLE on this hardware.")
    else:
        lines.append(f"  TTFT:            {t1['ttft_ms']:.0f} ms")
        lines.append(f"  Throughput:      {t1['tokens_per_second']:.2f} tokens/sec")
    lines.append(f"  PCIe crossings:  {t1['pcie_crossings']}")
    lines.append("")

    # Tier 2: GGML + Zig-Zag
    t2 = tiers["ggml_zigzag"]
    lines.append("--- TIER 2: GGML + FlexGen Zig-Zag (INT4) ---")
    lines.append(f"  Path:            SSD -> RAM buffer -> GPU VRAM -> Compute")
    lines.append(f"  Model on disk:   {t2['model_on_disk_gb']} GB")
    lines.append(f"  RAM needed:      {t2['ram_needed_gb']} GB (zig-zag buffer)")
    lines.append(f"  VRAM needed:     {t2['vram_needed_gb'] * 1024:.0f} MB (layer buffer)")
    if not t2["feasible"]:
        lines.append(f"  FATAL: {t2['fatal_reason']}")
    else:
        lines.append(f"  Per-layer I/O:   {t2['per_layer_io_ms']:.2f} ms (SSD + PCIe)")
        lines.append(f"  Per-layer compute: {t2['per_layer_compute_ms']:.4f} ms")
        lines.append(f"  Pipeline:        {'I/O-bound' if t2['bottleneck'] != 'compute' else 'GPU-bound'}")
        lines.append(f"  GPU utilization: {t2['gpu_utilization'] * 100:.1f}%")
        lines.append(f"  Forward pass:    {t2['forward_pass_ms']:.0f} ms")
        lines.append(f"  Throughput:      {t2['tokens_per_second']:.4f} tokens/sec")
    lines.append(f"  PCIe crossings:  {t2['pcie_crossings']}")
    lines.append("")

    # Tier 3: DirectStorage
    t3 = tiers["directstorage"]
    lines.append("--- TIER 3: DirectStorage (INT4 + GDeflate) ---")
    lines.append(f"  Path:            SSD -> GPU VRAM -> Compute (direct)")
    lines.append(f"  Model on disk:   {t3['model_on_disk_gb']} GB (GDeflate compressed)")
    lines.append(f"  RAM needed:      {t3['ram_needed_gb']:.2f} GB (staging buffer only)")
    lines.append(f"  VRAM needed:     {t3['vram_needed_gb'] * 1024:.0f} MB (ping-pong buffers)")
    if not t3["feasible"]:
        lines.append(f"  FATAL: {t3['fatal_reason']}")
    else:
        lines.append(f"  Per-layer I/O:   {t3['per_layer_io_ms']:.2f} ms (SSD -> GPU direct)")
        lines.append(f"  Per-layer compute: {t3['per_layer_compute_ms']:.4f} ms")
        lines.append(f"  Pipeline:        {'I/O-bound' if t3['bottleneck'] != 'compute' else 'GPU-bound'}")
        lines.append(f"  GPU utilization: {t3['gpu_utilization'] * 100:.1f}%")
        lines.append(f"  Forward pass:    {t3['forward_pass_ms']:.0f} ms")
        lines.append(f"  Throughput:      {t3['tokens_per_second']:.4f} tokens/sec")
    lines.append(f"  PCIe crossings:  {t3['pcie_crossings']}")
    lines.append("")

    # Comparison
    lines.append("--- COMPARISON ---")
    header = f"  {'Metric':<20} {'Tier 1':>12} {'Tier 2':>12} {'Tier 3':>12}"
    lines.append(header)
    sep = f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}"
    lines.append(sep)

    def _val(t, key, fmt=".1f", unit=""):
        v = t[key]
        if isinstance(v, bool):
            return "YES" if v else "NO"
        if isinstance(v, str):
            return v
        return f"{v:{fmt}}{unit}"

    lines.append(f"  {'Model on disk':<20} {_val(t1, 'model_on_disk_gb', unit=' GB'):>12} {_val(t2, 'model_on_disk_gb', unit=' GB'):>12} {_val(t3, 'model_on_disk_gb', unit=' GB'):>12}")
    lines.append(f"  {'RAM needed':<20} {_val(t1, 'ram_needed_gb', unit=' GB'):>12} {_val(t2, 'ram_needed_gb', unit=' GB'):>12} {t3['ram_needed_gb']:.2f} GB".rstrip())

    t1_vram = f"{t1['vram_needed_gb']:.1f} GB"
    t2_vram = f"{t2['vram_needed_gb'] * 1024:.0f} MB"
    t3_vram = f"{t3['vram_needed_gb'] * 1024:.0f} MB"
    lines.append(f"  {'VRAM needed':<20} {t1_vram:>12} {t2_vram:>12} {t3_vram:>12}")

    lines.append(f"  {'Feasible':<20} {_val(t1, 'feasible'):>12} {_val(t2, 'feasible'):>12} {_val(t3, 'feasible'):>12}")
    lines.append(f"  {'PCIe crossings':<20} {t1['pcie_crossings']:>12} {t2['pcie_crossings']:>12} {t3['pcie_crossings']:>12}")

    t1_tps = "N/A" if not t1["feasible"] else f"{t1['tokens_per_second']:.4f}"
    t2_tps = "N/A" if not t2["feasible"] else f"{t2['tokens_per_second']:.4f}"
    t3_tps = "N/A" if not t3["feasible"] else f"{t3['tokens_per_second']:.4f}"
    lines.append(f"  {'Throughput (tok/s)':<20} {t1_tps:>12} {t2_tps:>12} {t3_tps:>12}")

    speedup = comparison["speedup_ds_vs_zigzag"]
    if speedup > 0:
        lines.append(f"  {'Speedup vs Tier 2':<20} {'N/A':>12} {'1.0x':>12} {speedup:.1f}x".rstrip())
    lines.append("")

    winner = comparison["winner"]
    if winner:
        tier_names = {"pytorch": "Tier 1 (PyTorch)", "ggml_zigzag": "Tier 2 (GGML+ZigZag)", "directstorage": "Tier 3 (DirectStorage)"}
        lines.append(f"  WINNER: {tier_names.get(winner, winner)}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
