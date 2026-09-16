"""Hot/cold neuron segregation and slab-packed weight index.

Solves the Random I/O Trap: fetching 2,459 scattered 8KB neurons at random
hits ~150 MB/s on NVMe (130ms+ latency). Grouping co-activating neurons into
contiguous 2MB slabs keeps the SSD in sequential-read mode (7,000 MB/s),
dropping that to ~5ms — well within the 80ms draft window.

Offline profiling builds a co-activation graph, then graph-partitions neurons
into slabs. At runtime, the CUP predictor maps neuron IDs to slab IDs and
fetches whole slabs — intentionally loading some unneeded neurons to protect
CPU latency.

The hot/cold split is based on activation frequency across a profiling corpus:
  - Hot (~20%): fire chaotically across all contexts → lock in RAM permanently
  - Cold (~80%): fire topically, high temporal locality → leave on SSD, fetch via slabs
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NeuronMapConfig:
    n_neurons: int = 32000
    hot_ratio: float = 0.20
    slab_size_bytes: int = 2 * 1024 * 1024  # 2 MB
    neuron_size_bytes: int = 8 * 1024        # 8 KB per neuron
    over_provision_threshold: float = 0.35


@dataclass
class SlabIndex:
    """Maps cold neuron IDs to their host slab, built from co-activation profiling.

    Each slab is a contiguous block on disk containing neurons that tend to
    fire together. Fetching a slab loads all its neurons in one sequential read.
    """

    neuron_to_slab: np.ndarray      # [N_COLD] -> slab_id
    slab_to_neurons: dict[int, np.ndarray]  # slab_id -> array of cold neuron indices
    n_slabs: int
    neurons_per_slab: int

    @classmethod
    def from_coactivation(
        cls,
        coactivation_counts: np.ndarray,
        config: NeuronMapConfig,
    ) -> SlabIndex:
        """Build a slab index from a co-activation count matrix.

        Uses greedy graph partitioning: seed each slab with the most
        frequently co-activating pair, then fill to capacity with the
        neurons most correlated with the slab's current members.
        """
        n_cold = coactivation_counts.shape[0]
        neurons_per_slab = config.slab_size_bytes // config.neuron_size_bytes

        neuron_to_slab = np.full(n_cold, -1, dtype=np.int32)
        slab_to_neurons: dict[int, list[int]] = {}
        assigned = set()
        slab_id = 0

        # Greedy co-activation clustering
        while len(assigned) < n_cold:
            # Find the unassigned neuron with highest total co-activation
            remaining = [i for i in range(n_cold) if i not in assigned]
            if not remaining:
                break

            scores = coactivation_counts[remaining].sum(axis=1)
            seed = remaining[int(np.argmax(scores))]

            slab_members = [seed]
            assigned.add(seed)

            while len(slab_members) < neurons_per_slab and len(assigned) < n_cold:
                # Score unassigned neurons by co-activation with current slab members
                candidates = [i for i in range(n_cold) if i not in assigned]
                if not candidates:
                    break
                affinity = coactivation_counts[candidates][:, slab_members].sum(axis=1)
                best = candidates[int(np.argmax(affinity))]
                slab_members.append(best)
                assigned.add(best)

            for neuron in slab_members:
                neuron_to_slab[neuron] = slab_id
            slab_to_neurons[slab_id] = np.array(slab_members, dtype=np.int32)
            slab_id += 1

        return cls(
            neuron_to_slab=neuron_to_slab,
            slab_to_neurons=slab_to_neurons,
            n_slabs=slab_id,
            neurons_per_slab=neurons_per_slab,
        )

    @classmethod
    def uniform(cls, n_cold: int, config: NeuronMapConfig) -> SlabIndex:
        """Simple uniform partitioning (no profiling data). Baseline for comparison."""
        neurons_per_slab = config.slab_size_bytes // config.neuron_size_bytes
        neuron_to_slab = np.zeros(n_cold, dtype=np.int32)
        slab_to_neurons: dict[int, np.ndarray] = {}

        slab_id = 0
        for start in range(0, n_cold, neurons_per_slab):
            end = min(start + neurons_per_slab, n_cold)
            members = np.arange(start, end, dtype=np.int32)
            for idx in members:
                neuron_to_slab[idx] = slab_id
            slab_to_neurons[slab_id] = members
            slab_id += 1

        return cls(
            neuron_to_slab=neuron_to_slab,
            slab_to_neurons=slab_to_neurons,
            n_slabs=slab_id,
            neurons_per_slab=neurons_per_slab,
        )

    def neurons_to_slabs(self, neuron_ids: np.ndarray) -> np.ndarray:
        """Map a set of cold neuron IDs to the unique slab IDs that cover them."""
        slab_ids = self.neuron_to_slab[neuron_ids]
        return np.unique(slab_ids)

    def expand_slabs(self, slab_ids: np.ndarray) -> np.ndarray:
        """Expand slab IDs back to all neuron IDs contained in those slabs."""
        neurons = []
        for sid in slab_ids:
            if sid in self.slab_to_neurons:
                neurons.append(self.slab_to_neurons[sid])
        if neurons:
            return np.unique(np.concatenate(neurons))
        return np.array([], dtype=np.int32)

    def fetch_stats(self, requested_neurons: np.ndarray) -> dict:
        """Compute I/O statistics for a fetch request."""
        slab_ids = self.neurons_to_slabs(requested_neurons)
        all_loaded = self.expand_slabs(slab_ids)
        requested_set = set(requested_neurons)
        wasted = len(all_loaded) - len(requested_set & set(all_loaded))

        n_slabs = len(slab_ids)
        total_bytes = n_slabs * (self.neurons_per_slab * 8 * 1024)  # approx
        seq_time_ms = (total_bytes / (7_000 * 1024 * 1024)) * 1000
        rand_time_ms = (len(requested_neurons) * 8 * 1024 / (150 * 1024 * 1024)) * 1000

        return {
            "n_neurons_requested": len(requested_neurons),
            "n_slabs_fetched": n_slabs,
            "n_neurons_loaded": len(all_loaded),
            "n_neurons_wasted": wasted,
            "total_bytes_mb": round(total_bytes / (1024 * 1024), 2),
            "sequential_read_ms": round(seq_time_ms, 2),
            "random_read_ms": round(rand_time_ms, 2),
            "speedup": round(rand_time_ms / seq_time_ms, 1) if seq_time_ms > 0 else float("inf"),
        }


@dataclass
class NeuronMap:
    """Complete hot/cold neuron map for a model layer.

    Hot neurons are indexed by their positions (always in RAM).
    Cold neurons are indexed through the SlabIndex (fetched from SSD).
    """

    n_neurons: int
    hot_mask: np.ndarray        # [N_NEURONS] bool — True for hot neurons
    cold_mask: np.ndarray       # [N_NEURONS] bool — True for cold neurons
    hot_indices: np.ndarray     # sorted indices of hot neurons
    cold_indices: np.ndarray    # sorted indices of cold neurons
    slab_index: SlabIndex
    config: NeuronMapConfig

    @classmethod
    def from_activation_frequencies(
        cls,
        frequencies: np.ndarray,
        config: NeuronMapConfig,
        coactivation_counts: np.ndarray | None = None,
    ) -> NeuronMap:
        """Build a neuron map from per-neuron activation frequencies.

        The top `hot_ratio` fraction by frequency become hot (locked in RAM).
        The rest are cold (on SSD), partitioned into slabs.
        """
        n = len(frequencies)
        n_hot = int(n * config.hot_ratio)

        hot_indices = np.argsort(frequencies)[-n_hot:]
        hot_indices = np.sort(hot_indices)
        hot_mask = np.zeros(n, dtype=bool)
        hot_mask[hot_indices] = True

        cold_mask = ~hot_mask
        cold_indices = np.where(cold_mask)[0]

        if coactivation_counts is not None:
            slab_index = SlabIndex.from_coactivation(coactivation_counts, config)
        else:
            slab_index = SlabIndex.uniform(len(cold_indices), config)

        return cls(
            n_neurons=n,
            hot_mask=hot_mask,
            cold_mask=cold_mask,
            hot_indices=hot_indices,
            cold_indices=cold_indices,
            slab_index=slab_index,
            config=config,
        )

    @property
    def n_hot(self) -> int:
        return len(self.hot_indices)

    @property
    def n_cold(self) -> int:
        return len(self.cold_indices)

    def cold_to_local(self, global_neuron_ids: np.ndarray) -> np.ndarray:
        """Map global neuron IDs to cold-local indices (for slab lookup)."""
        # cold_indices is sorted, so searchsorted gives the local position
        return np.searchsorted(self.cold_indices, global_neuron_ids)

    def local_to_cold(self, local_ids: np.ndarray) -> np.ndarray:
        """Map cold-local indices back to global neuron IDs."""
        return self.cold_indices[local_ids]
