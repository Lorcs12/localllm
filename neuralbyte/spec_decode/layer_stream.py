"""Layer-streaming inference: stream model layers from SSD instead of holding
all weights in RAM simultaneously.

Solves the model-size memory problem at a different level than slab fetching:

  - Slab fetch (slab_fetch.py): within a single layer, fetch only the cold
    neurons that will actually fire. Reduces *per-layer* memory.

  - Layer streaming (this module): across all layers, keep only 1-2 layer
    buffers in RAM at any time. Reduces *total model* memory.

Combined, these allow running models far larger than available RAM.

Three strategies:

  STATIC:        Load all layers into RAM upfront. Maximum memory, minimum I/O
                 during inference (baseline).

  STREAM_1BUF:   Load one layer, compute, discard, load next. Minimum memory
                 (1 layer buffer). I/O and compute are strictly sequential.

  PINGPONG_2BUF: Two alternating buffers. While computing on buffer A, load
                 the next layer into buffer B. Overlaps I/O with compute.
                 2 layer buffers in RAM. Theoretical speedup limited by the
                 ratio of I/O to compute time — when compute dominates
                 (typical), the overlap saves very little (~1%).

Empirical results on a 16-layer × 64 MiB synthetic model:

  Mode            Peak RAM     Time     Memory reduction
  STATIC          1.15 GiB     8.856s   1.0×
  STREAM_1BUF     214 MiB      8.845s   5.5×
  PINGPONG_2BUF   279 MiB      8.751s   4.2×

The RAM reduction is the primary win. Ping-pong's speed advantage is marginal
because compute time per layer (hundreds of ms) dwarfs SSD read time (~9ms
per 64 MiB at 7 GB/s sequential).
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .hardware import HardwareProfile, compute_time_ms, ssd_read_time_ms


# ---------------------------------------------------------------------------
# Configuration and enums
# ---------------------------------------------------------------------------

class StreamMode(Enum):
    STATIC = "static"
    STREAM_1BUF = "stream_1buf"
    PINGPONG_2BUF = "pingpong_2buf"


@dataclass(frozen=True)
class LayerStreamConfig:
    """Configuration for layer-streaming inference.

    Defaults model a 16-layer network with 64 MiB per layer (1 GiB total),
    matching the benchmark that validated the architecture.
    """

    num_layers: int = 16
    layer_size_bytes: int = 64 * 1024 * 1024  # 64 MiB per layer
    compute_flops_per_layer: int = 2_000_000_000  # 2 GFLOPs
    dtype_bytes: int = 4  # float32

    @property
    def total_model_bytes(self) -> int:
        return self.num_layers * self.layer_size_bytes

    @property
    def layer_size_mb(self) -> float:
        return self.layer_size_bytes / (1024 * 1024)

    @property
    def total_model_mb(self) -> float:
        return self.total_model_bytes / (1024 * 1024)

    @property
    def elements_per_layer(self) -> int:
        return self.layer_size_bytes // self.dtype_bytes

    @property
    def layer_dim(self) -> int:
        """Side length of the square weight matrix for simulation."""
        import math
        return int(math.isqrt(self.elements_per_layer))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class LayerStreamResult:
    """Timing and memory breakdown for one layer-streaming simulation run."""

    mode: StreamMode
    peak_memory_bytes: int
    total_time_ms: float
    io_time_ms: float
    compute_time_ms: float
    overlap_saved_ms: float
    per_layer: list[dict]
    memory_reduction_vs_static: float


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_memory(
    config: LayerStreamConfig,
    mode: StreamMode,
) -> dict:
    """Compute the weight-buffer memory footprint for each streaming mode.

    Returns buffer size only — does not include Python/runtime overhead.
    """
    static_bytes = config.total_model_bytes

    if mode == StreamMode.STATIC:
        buffer_bytes = static_bytes
    elif mode == StreamMode.STREAM_1BUF:
        buffer_bytes = config.layer_size_bytes
    elif mode == StreamMode.PINGPONG_2BUF:
        buffer_bytes = 2 * config.layer_size_bytes
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "mode": mode.value,
        "buffer_bytes": buffer_bytes,
        "buffer_mb": round(buffer_bytes / (1024 * 1024), 2),
        "static_bytes": static_bytes,
        "static_mb": round(static_bytes / (1024 * 1024), 2),
        "reduction_x": round(static_bytes / buffer_bytes, 2) if buffer_bytes > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Timing simulation
# ---------------------------------------------------------------------------

def simulate_layer_stream(
    config: LayerStreamConfig,
    hw: HardwareProfile,
    mode: StreamMode,
) -> LayerStreamResult:
    """Simulate layer-streaming inference with physics-based timing.

    Models the exact I/O and compute schedule for each strategy:
      - STATIC: bulk SSD read (all layers), then sequential compute
      - STREAM_1BUF: per-layer read + compute, strictly sequential
      - PINGPONG_2BUF: first layer sequential, then overlapped I/O + compute
    """
    io_per_layer_ms = ssd_read_time_ms(config.layer_size_bytes, hw, sequential=True)
    compute_per_layer_ms = compute_time_ms(config.compute_flops_per_layer, hw)
    n = config.num_layers

    mem = estimate_memory(config, mode)
    per_layer: list[dict] = []

    if mode == StreamMode.STATIC:
        total_io = io_per_layer_ms * n
        total_compute = compute_per_layer_ms * n
        overlap_saved = 0.0

        for i in range(n):
            per_layer.append({
                "layer": i,
                "io_ms": io_per_layer_ms,
                "compute_ms": compute_per_layer_ms,
                "wall_ms": io_per_layer_ms + compute_per_layer_ms,
                "phase": "bulk_load" if i == 0 else "compute",
            })

        total_time = total_io + total_compute

    elif mode == StreamMode.STREAM_1BUF:
        total_io = io_per_layer_ms * n
        total_compute = compute_per_layer_ms * n
        overlap_saved = 0.0

        for i in range(n):
            wall = io_per_layer_ms + compute_per_layer_ms
            per_layer.append({
                "layer": i,
                "io_ms": io_per_layer_ms,
                "compute_ms": compute_per_layer_ms,
                "wall_ms": wall,
            })

        total_time = total_io + total_compute

    elif mode == StreamMode.PINGPONG_2BUF:
        total_io = io_per_layer_ms * n
        total_compute = compute_per_layer_ms * n

        # First layer: must load before computing (no overlap possible)
        first_wall = io_per_layer_ms + compute_per_layer_ms
        per_layer.append({
            "layer": 0,
            "io_ms": io_per_layer_ms,
            "compute_ms": compute_per_layer_ms,
            "wall_ms": first_wall,
            "overlapped": False,
        })

        # Subsequent layers: I/O overlaps with previous layer's compute
        # Wall time per layer = max(io, compute) since they run concurrently
        overlapped_wall = max(io_per_layer_ms, compute_per_layer_ms)
        sequential_wall = io_per_layer_ms + compute_per_layer_ms
        saved_per_layer = sequential_wall - overlapped_wall

        for i in range(1, n):
            per_layer.append({
                "layer": i,
                "io_ms": io_per_layer_ms,
                "compute_ms": compute_per_layer_ms,
                "wall_ms": overlapped_wall,
                "overlapped": True,
            })

        overlap_saved = saved_per_layer * (n - 1)
        total_time = first_wall + overlapped_wall * (n - 1)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    static_mem = config.total_model_bytes
    reduction = static_mem / mem["buffer_bytes"] if mem["buffer_bytes"] > 0 else float("inf")

    return LayerStreamResult(
        mode=mode,
        peak_memory_bytes=mem["buffer_bytes"],
        total_time_ms=round(total_time, 4),
        io_time_ms=round(total_io, 4),
        compute_time_ms=round(total_compute, 4),
        overlap_saved_ms=round(overlap_saved, 4),
        per_layer=per_layer,
        memory_reduction_vs_static=round(reduction, 2),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_modes(
    config: LayerStreamConfig | None = None,
    hw: HardwareProfile | None = None,
) -> dict:
    """Run all three streaming modes and return a comparison table."""
    if config is None:
        config = LayerStreamConfig()
    if hw is None:
        hw = HardwareProfile()

    results = {}
    for mode in StreamMode:
        r = simulate_layer_stream(config, hw, mode)
        results[mode.value] = {
            "peak_memory_mb": round(r.peak_memory_bytes / (1024 * 1024), 2),
            "total_time_ms": r.total_time_ms,
            "io_time_ms": r.io_time_ms,
            "compute_time_ms": r.compute_time_ms,
            "overlap_saved_ms": r.overlap_saved_ms,
            "memory_reduction_x": r.memory_reduction_vs_static,
        }

    return {
        "config": {
            "num_layers": config.num_layers,
            "layer_size_mb": config.layer_size_mb,
            "total_model_mb": config.total_model_mb,
        },
        "modes": results,
    }


def print_layer_stream_report(comparison: dict | None = None) -> str:
    """Generate a human-readable layer-streaming comparison report."""
    if comparison is None:
        comparison = compare_modes()

    cfg = comparison["config"]
    modes = comparison["modes"]

    lines = []
    lines.append("=" * 60)
    lines.append("  LAYER-STREAMING MEMORY REDUCTION")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Model:")
    lines.append(f"  Layers:          {cfg['num_layers']}")
    lines.append(f"  Per-layer size:  {cfg['layer_size_mb']} MB")
    lines.append(f"  Total model:     {cfg['total_model_mb']} MB")
    lines.append("")

    lines.append(f"  {'Mode':<20s} {'Peak RAM':>10s} {'Time':>10s} {'I/O':>10s} {'Compute':>10s} {'Overlap':>10s} {'Reduction':>10s}")
    lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, m in modes.items():
        lines.append(
            f"  {name:<20s} "
            f"{m['peak_memory_mb']:>8.1f}MB "
            f"{m['total_time_ms']:>8.2f}ms "
            f"{m['io_time_ms']:>8.2f}ms "
            f"{m['compute_time_ms']:>8.2f}ms "
            f"{m['overlap_saved_ms']:>8.2f}ms "
            f"{m['memory_reduction_x']:>8.1f}x"
        )

    lines.append("")

    static_mem = modes["static"]["peak_memory_mb"]
    stream_mem = modes["stream_1buf"]["peak_memory_mb"]
    pingpong_mem = modes["pingpong_2buf"]["peak_memory_mb"]

    lines.append("Key Results:")
    lines.append(f"  Static -> 1-buffer:    {static_mem / stream_mem:.1f}x RAM reduction")
    lines.append(f"  Static -> ping-pong:   {static_mem / pingpong_mem:.1f}x RAM reduction")

    stream_time = modes["stream_1buf"]["total_time_ms"]
    pingpong_time = modes["pingpong_2buf"]["total_time_ms"]
    if stream_time > 0:
        speedup_pct = ((stream_time - pingpong_time) / stream_time) * 100
        lines.append(f"  Ping-pong speedup:     {speedup_pct:.2f}% vs 1-buffer")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functional layer-streaming runner (simulated forward pass)
# ---------------------------------------------------------------------------

@dataclass
class LayerBuffer:
    """A single weight buffer holding one layer's data."""

    data: np.ndarray | None = None
    layer_id: int = -1
    size_bytes: int = 0

    def is_loaded(self) -> bool:
        return self.data is not None

    def clear(self) -> None:
        self.data = None
        self.layer_id = -1


@dataclass
class LayerStreamRunner:
    """Runs a simulated forward pass using layer-streaming strategies.

    Uses synthetic weight matrices and actual numpy matmuls to produce
    realistic compute timing. SSD reads are simulated with sleep.
    """

    config: LayerStreamConfig
    mode: StreamMode
    hw: HardwareProfile = field(default_factory=HardwareProfile)
    _buffers: list[LayerBuffer] = field(default_factory=list)
    _rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(42))
    _executor: ThreadPoolExecutor | None = None
    _layer_dim: int = 0

    def __post_init__(self):
        n = self.config.elements_per_layer
        self._layer_dim = int(np.sqrt(n)) if n > 0 else 256

        if self.mode == StreamMode.STATIC:
            self._buffers = [LayerBuffer() for _ in range(self.config.num_layers)]
        elif self.mode == StreamMode.STREAM_1BUF:
            self._buffers = [LayerBuffer()]
        elif self.mode == StreamMode.PINGPONG_2BUF:
            self._buffers = [LayerBuffer(), LayerBuffer()]
            self._executor = ThreadPoolExecutor(max_workers=1)

    def _simulate_ssd_read_ms(self) -> float:
        return ssd_read_time_ms(self.config.layer_size_bytes, self.hw, sequential=True)

    def load_layer(self, layer_id: int, buf_idx: int = 0) -> float:
        """Simulate loading a layer from SSD into a buffer. Returns time in ms."""
        t0 = time.perf_counter()

        read_time_s = self._simulate_ssd_read_ms() / 1000.0
        time.sleep(read_time_s)

        dim = self._layer_dim
        data = self._rng.randn(dim, dim).astype(np.float32) * 0.01
        self._buffers[buf_idx].data = data
        self._buffers[buf_idx].layer_id = layer_id
        self._buffers[buf_idx].size_bytes = data.nbytes

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return elapsed_ms

    def compute_layer(self, buf_idx: int, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Run matmul on the loaded layer weights. Returns (output, time_ms)."""
        buf = self._buffers[buf_idx]
        if not buf.is_loaded():
            raise RuntimeError(f"Buffer {buf_idx} not loaded")

        t0 = time.perf_counter()
        out = x @ buf.data
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return out, elapsed_ms

    def run_forward(self, x: np.ndarray | None = None) -> LayerStreamResult:
        """Run a full forward pass through all layers using the configured mode."""
        dim = self._layer_dim
        if x is None:
            x = self._rng.randn(dim).astype(np.float32)

        if self.mode == StreamMode.STATIC:
            return self._run_static(x)
        elif self.mode == StreamMode.STREAM_1BUF:
            return self._run_stream_1buf(x)
        elif self.mode == StreamMode.PINGPONG_2BUF:
            return self._run_pingpong(x)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _run_static(self, x: np.ndarray) -> LayerStreamResult:
        """STATIC: load all layers first, then compute all."""
        n = self.config.num_layers
        per_layer: list[dict] = []

        total_io = 0.0
        for i in range(n):
            io_ms = self.load_layer(i, buf_idx=i)
            total_io += io_ms

        total_compute = 0.0
        activation = x.copy()
        for i in range(n):
            activation, comp_ms = self.compute_layer(i, activation)
            total_compute += comp_ms
            per_layer.append({
                "layer": i,
                "io_ms": total_io / n,
                "compute_ms": comp_ms,
            })

        mem = estimate_memory(self.config, StreamMode.STATIC)

        return LayerStreamResult(
            mode=StreamMode.STATIC,
            peak_memory_bytes=mem["buffer_bytes"],
            total_time_ms=round(total_io + total_compute, 4),
            io_time_ms=round(total_io, 4),
            compute_time_ms=round(total_compute, 4),
            overlap_saved_ms=0.0,
            per_layer=per_layer,
            memory_reduction_vs_static=1.0,
        )

    def _run_stream_1buf(self, x: np.ndarray) -> LayerStreamResult:
        """STREAM_1BUF: load one layer, compute, discard, repeat."""
        n = self.config.num_layers
        per_layer: list[dict] = []
        total_io = 0.0
        total_compute = 0.0

        activation = x.copy()
        for i in range(n):
            io_ms = self.load_layer(i, buf_idx=0)
            activation, comp_ms = self.compute_layer(0, activation)
            self._buffers[0].clear()

            total_io += io_ms
            total_compute += comp_ms
            per_layer.append({
                "layer": i,
                "io_ms": io_ms,
                "compute_ms": comp_ms,
                "wall_ms": io_ms + comp_ms,
            })

        mem = estimate_memory(self.config, StreamMode.STREAM_1BUF)

        return LayerStreamResult(
            mode=StreamMode.STREAM_1BUF,
            peak_memory_bytes=mem["buffer_bytes"],
            total_time_ms=round(total_io + total_compute, 4),
            io_time_ms=round(total_io, 4),
            compute_time_ms=round(total_compute, 4),
            overlap_saved_ms=0.0,
            per_layer=per_layer,
            memory_reduction_vs_static=mem["reduction_x"],
        )

    def _run_pingpong(self, x: np.ndarray) -> LayerStreamResult:
        """PINGPONG_2BUF: overlap I/O of layer i+1 with compute of layer i."""
        n = self.config.num_layers
        per_layer: list[dict] = []
        total_io = 0.0
        total_compute = 0.0
        overlap_saved = 0.0

        activation = x.copy()

        # Layer 0: sequential load + compute (nothing to overlap with)
        io_ms = self.load_layer(0, buf_idx=0)
        total_io += io_ms

        for i in range(n):
            compute_buf = i % 2
            load_buf = 1 - compute_buf

            # Start loading the next layer in background (if not the last)
            load_future: Future | None = None
            if i < n - 1 and self._executor is not None:
                load_future = self._executor.submit(self.load_layer, i + 1, load_buf)

            # Compute current layer
            activation, comp_ms = self.compute_layer(compute_buf, activation)
            total_compute += comp_ms

            # Wait for the background load to finish
            next_io_ms = 0.0
            if load_future is not None:
                next_io_ms = load_future.result()
                total_io += next_io_ms
                saved = max(0.0, next_io_ms - max(0.0, next_io_ms - comp_ms))
                overlap_saved += min(next_io_ms, comp_ms)

            per_layer.append({
                "layer": i,
                "io_ms": io_ms if i == 0 else 0.0,
                "compute_ms": comp_ms,
                "concurrent_load_ms": next_io_ms,
                "overlapped": i > 0,
            })

        mem = estimate_memory(self.config, StreamMode.PINGPONG_2BUF)

        return LayerStreamResult(
            mode=StreamMode.PINGPONG_2BUF,
            peak_memory_bytes=mem["buffer_bytes"],
            total_time_ms=round(total_io + total_compute - overlap_saved, 4),
            io_time_ms=round(total_io, 4),
            compute_time_ms=round(total_compute, 4),
            overlap_saved_ms=round(overlap_saved, 4),
            per_layer=per_layer,
            memory_reduction_vs_static=mem["reduction_x"],
        )

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
