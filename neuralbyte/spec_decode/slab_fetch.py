"""Async slab fetcher — concurrent SSD reads while the draft head generates tokens.

The key latency-hiding trick: Thread A (CPU) runs the SMW draft head to
generate K tokens (~80ms). Thread B (SSD) simultaneously fetches the cold
neuron slabs the CUP predictor identified (~5ms for sequential 2MB slab reads,
vs ~130ms for random 8KB neuron reads). Thread A finishes first; the slabs
are already in RAM for the frontier verification pass.

Slab fetching turns the I/O pattern from random reads (150 MB/s) to sequential
reads (7,000 MB/s) by trading bandwidth for latency: we load whole 2MB slabs
even though some neurons in each slab aren't needed. The ~50% bandwidth
overhead buys a 25x latency reduction.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .neuron_map import NeuronMap, SlabIndex
from .predictor import PredictionResult


@dataclass
class FetchResult:
    """Result of an async slab fetch operation."""

    slab_ids: np.ndarray            # which slabs were fetched
    neuron_ids: np.ndarray          # all neuron IDs loaded (including over-provisioned)
    weights: dict[int, np.ndarray]  # slab_id -> weight data
    fetch_time_ms: float
    n_slabs: int
    n_neurons_loaded: int
    total_bytes: int
    sequential: bool                # True if slab-based, False if random


@dataclass
class SlabFetcher:
    """Fetches cold neuron weight slabs from SSD concurrently with token drafting.

    In production, `slab_dir` points to the directory of pre-packed slab files
    (each exactly 2MB). For simulation, `SlabFetcher.simulated()` creates a
    fetcher that returns synthetic weight data with realistic timing.
    """

    neuron_map: NeuronMap
    slab_dir: Path | None = None
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=2))
    _simulated: bool = False
    _sim_neuron_dim: int = 0
    _sim_rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(0))

    @classmethod
    def simulated(
        cls,
        neuron_map: NeuronMap,
        neuron_dim: int = 8192,
        seed: int = 0,
    ) -> SlabFetcher:
        """Create a simulated fetcher that generates synthetic weight data.

        Simulates realistic SSD timing: ~0.7ms per slab (sequential) vs
        ~0.05ms per neuron (random). These match NVMe 7 GB/s sequential
        and 150 MB/s random at 8KB per neuron.
        """
        return cls(
            neuron_map=neuron_map,
            slab_dir=None,
            _simulated=True,
            _sim_neuron_dim=neuron_dim,
            _sim_rng=np.random.RandomState(seed),
        )

    def fetch_slabs_async(self, prediction: PredictionResult) -> Future[FetchResult]:
        """Submit an async slab fetch — runs on Thread B while Thread A drafts.

        Maps predicted cold neuron IDs to their host slabs, then fetches
        whole slabs. Returns a Future that resolves when all slabs are in RAM.
        """
        cold_local_ids = prediction.predicted_cold_ids
        return self._executor.submit(self._fetch_slabs, cold_local_ids)

    def fetch_slabs_sync(self, prediction: PredictionResult) -> FetchResult:
        """Synchronous slab fetch — for testing or when concurrency isn't needed."""
        return self._fetch_slabs(prediction.predicted_cold_ids)

    def _fetch_slabs(self, cold_local_ids: np.ndarray) -> FetchResult:
        """Fetch all slabs containing the requested cold neurons."""
        t0 = time.perf_counter()

        slab_index = self.neuron_map.slab_index
        slab_ids = slab_index.neurons_to_slabs(cold_local_ids)
        all_neuron_ids = slab_index.expand_slabs(slab_ids)

        weights: dict[int, np.ndarray] = {}
        total_bytes = 0

        for sid in slab_ids:
            slab_neurons = slab_index.slab_to_neurons[sid]
            n_neurons = len(slab_neurons)

            if self._simulated:
                # Simulate sequential read timing: 2MB / 7GB/s ≈ 0.28ms per slab
                time.sleep(0.00028)
                data = self._sim_rng.randn(n_neurons, self._sim_neuron_dim).astype(np.float32)
            elif self.slab_dir is not None:
                slab_path = self.slab_dir / f"slab_{sid:06d}.bin"
                data = np.fromfile(slab_path, dtype=np.float32).reshape(n_neurons, -1)
            else:
                data = np.zeros((n_neurons, 1), dtype=np.float32)

            weights[int(sid)] = data
            total_bytes += data.nbytes

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FetchResult(
            slab_ids=slab_ids,
            neuron_ids=all_neuron_ids,
            weights=weights,
            fetch_time_ms=elapsed_ms,
            n_slabs=len(slab_ids),
            n_neurons_loaded=len(all_neuron_ids),
            total_bytes=total_bytes,
            sequential=True,
        )

    def fetch_random_async(self, cold_local_ids: np.ndarray) -> Future[FetchResult]:
        """Baseline: fetch individual neurons randomly (no slab packing).

        For comparison — shows the random I/O penalty that slab packing avoids.
        """
        return self._executor.submit(self._fetch_random, cold_local_ids)

    def _fetch_random(self, cold_local_ids: np.ndarray) -> FetchResult:
        """Fetch neurons individually — random I/O baseline."""
        t0 = time.perf_counter()

        weights: dict[int, np.ndarray] = {}
        total_bytes = 0

        for nid in cold_local_ids:
            if self._simulated:
                # Simulate random read: 8KB / 150MB/s ≈ 0.052ms per neuron
                time.sleep(0.000052)
                data = self._sim_rng.randn(1, self._sim_neuron_dim).astype(np.float32)
            else:
                data = np.zeros((1, 1), dtype=np.float32)

            weights[int(nid)] = data
            total_bytes += data.nbytes

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FetchResult(
            slab_ids=np.array([], dtype=np.int32),
            neuron_ids=cold_local_ids,
            weights=weights,
            fetch_time_ms=elapsed_ms,
            n_slabs=0,
            n_neurons_loaded=len(cold_local_ids),
            total_bytes=total_bytes,
            sequential=False,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


@dataclass
class PipelineTimer:
    """Records wall-clock timing for each phase of the concurrent pipeline."""

    _phases: list[dict] = field(default_factory=list)

    def record(self, phase: str, duration_ms: float, **kwargs: Any) -> None:
        entry = {"phase": phase, "duration_ms": round(duration_ms, 3)}
        entry.update(kwargs)
        self._phases.append(entry)

    def total_ms(self) -> float:
        return sum(p["duration_ms"] for p in self._phases)

    def to_dict(self) -> dict:
        phases = {p["phase"]: p["duration_ms"] for p in self._phases}
        phases["total"] = self.total_ms()
        return phases

    def summary(self) -> str:
        lines = [f"  {p['phase']:20s} {p['duration_ms']:8.2f} ms" for p in self._phases]
        lines.append(f"  {'TOTAL':20s} {self.total_ms():8.2f} ms")
        return "\n".join(lines)
