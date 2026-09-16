"""Sparse compute layer: fused gather-GEMM vs standard PyTorch indexing.

The Sparse Compute Penalty: when you run `W[active_indices] @ x` in PyTorch,
the runtime (1) allocates a new contiguous tensor, (2) copies scattered rows
into it, (3) then does the math. The memcpy takes 34x longer than the math.

The fix: a fused gather-GEMM kernel that reads directly from scattered memory
positions, multiplies in-register, and accumulates — no allocation, no copy.
In production this is a custom C++/CUDA/Metal kernel. Here we model the
latency to prove the pipeline math works, and provide a numpy reference
implementation of the fused scatter-gather pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hardware import HardwareProfile, compute_time_ms, memcpy_time_ms


@dataclass(frozen=True)
class SparseLayerConfig:
    """Configuration for one sparse FFN layer of the frontier model."""

    d_model: int = 8192
    ffn_dim: int = 28672
    sparsity: float = 0.85
    dtype_bytes: int = 2  # FP16

    @property
    def active_neurons(self) -> int:
        return int(self.ffn_dim * (1.0 - self.sparsity))


def sparse_matmul_standard(
    W: np.ndarray,
    x: np.ndarray,
    active_indices: np.ndarray,
) -> np.ndarray:
    """Standard PyTorch-style sparse matmul: index → copy → matmul.

    W: [ffn_dim, d_model], x: [d_model], active_indices: [n_active]
    This is what `W[active_indices] @ x` does under the hood.
    """
    active_weights = W[active_indices]  # allocate + copy
    return active_weights @ x           # math on the copy


def sparse_matmul_fused(
    W: np.ndarray,
    x: np.ndarray,
    active_indices: np.ndarray,
) -> np.ndarray:
    """Fused gather-GEMM: reads directly from scattered positions in W.

    No allocation, no copy. Each active row is read once, multiplied with x
    in-register, and the result accumulated. In production this is a C++
    kernel; here we simulate it with explicit row-by-row accumulation.
    """
    result = np.empty(len(active_indices), dtype=np.float64)
    for i, idx in enumerate(active_indices):
        result[i] = np.dot(W[idx], x)
    return result


def estimate_sparse_layer_time(
    config: SparseLayerConfig,
    hw: HardwareProfile,
    *,
    fused: bool = False,
) -> dict:
    """Estimate wall-clock time for one sparse FFN layer.

    Standard path: memcpy_time + math_time
    Fused path: math_time only (memcpy eliminated)
    """
    n_active = config.active_neurons
    flops = n_active * config.d_model * 2  # multiply + accumulate

    math_ms = compute_time_ms(flops, hw)
    copy_ms = memcpy_time_ms(n_active * config.d_model, config.dtype_bytes, hw)

    if fused:
        return {
            "strategy": "fused_gather_gemm",
            "copy_ms": 0.0,
            "math_ms": round(math_ms, 4),
            "total_ms": round(math_ms, 4),
            "active_neurons": n_active,
        }
    else:
        return {
            "strategy": "standard_pytorch",
            "copy_ms": round(copy_ms, 4),
            "math_ms": round(math_ms, 4),
            "total_ms": round(copy_ms + math_ms, 4),
            "active_neurons": n_active,
        }


def estimate_full_model_sparse_time(
    config: SparseLayerConfig,
    hw: HardwareProfile,
    num_layers: int = 80,
    *,
    fused: bool = False,
) -> dict:
    """Estimate total sparse FFN time across all layers of the frontier model."""
    per_layer = estimate_sparse_layer_time(config, hw, fused=fused)
    total_ms = per_layer["total_ms"] * num_layers
    copy_total = per_layer["copy_ms"] * num_layers
    math_total = per_layer["math_ms"] * num_layers

    return {
        "strategy": per_layer["strategy"],
        "num_layers": num_layers,
        "per_layer_ms": per_layer["total_ms"],
        "total_copy_ms": round(copy_total, 2),
        "total_math_ms": round(math_total, 2),
        "total_ms": round(total_ms, 2),
        "speedup_vs_standard": (
            round((copy_total + math_total) / math_total, 1)
            if fused and math_total > 0
            else None
        ),
    }


def verify_fused_correctness(
    d_model: int = 256,
    ffn_dim: int = 1024,
    n_active: int = 150,
    seed: int = 42,
) -> dict:
    """Verify that fused gather-GEMM produces identical results to standard."""
    rng = np.random.RandomState(seed)
    W = rng.randn(ffn_dim, d_model).astype(np.float64)
    x = rng.randn(d_model).astype(np.float64)
    indices = rng.choice(ffn_dim, n_active, replace=False).astype(np.int32)

    standard = sparse_matmul_standard(W, x, indices)
    fused = sparse_matmul_fused(W, x, indices)

    max_err = float(np.max(np.abs(standard - fused)))
    return {
        "max_error": max_err,
        "match": max_err < 1e-10,
        "n_active": n_active,
        "d_model": d_model,
    }
