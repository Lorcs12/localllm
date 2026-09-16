"""KV Cache management with quantization and heavy-hitter eviction.

Solves Bottleneck #3: attention is 0% sparse — every head must compare the
current token against every past token's Key and Value vectors. At 32K context,
a 70B model's KV cache is ~10 GB in FP16, taking 163ms to read through RAM
bandwidth. That single read blows the entire pipeline budget.

Two complementary fixes:

  1. KIVI-style quantization: compress KV entries from FP16 (2 bytes) to INT4
     (0.5 bytes), reducing cache size by 75%. Older tokens that no longer need
     fine-grained precision are quantized more aggressively.

  2. H2O/StreamingLLM-style eviction: not all past tokens matter equally.
     Attention scores follow a severe Pareto distribution:
       - Sink tokens (first ~4): absorb disproportionate attention mass
       - Local window (last ~256): the current working context
       - Heavy-hitters: a sparse set of high-attention tokens in the middle
     Everything else is evicted — the model never reads it again.

Combined, these reduce a 10 GB cache to ~0.12 GB, cutting RAM read latency
from 163ms to 2ms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hardware import HardwareProfile


@dataclass(frozen=True)
class KVCacheConfig:
    """Architecture constants for the frontier model's KV cache."""

    num_layers: int = 80
    kv_heads: int = 8
    head_dim: int = 128

    # Eviction policy
    n_sink_tokens: int = 4
    n_local_tokens: int = 256
    n_heavy_hitters: int = 1000

    # Quantization
    quant_bits: int = 4  # INT4

    @property
    def bytes_per_fp16(self) -> float:
        return 2.0

    @property
    def bytes_per_quantized(self) -> float:
        return self.quant_bits / 8.0

    @property
    def params_per_token(self) -> int:
        """KV parameters stored per token: 2 (K+V) x layers x heads x head_dim."""
        return 2 * self.num_layers * self.kv_heads * self.head_dim


@dataclass
class KVEntry:
    """One token's KV cache entry with attention score tracking."""

    position: int
    attention_score: float = 0.0
    is_sink: bool = False
    is_local: bool = False
    quantized: bool = False


@dataclass
class KVCache:
    """Managed KV cache with eviction and quantization policies.

    Tracks which tokens are retained, their attention importance, and their
    quantization state. Computes the actual memory footprint and RAM read
    latency at any point during generation.
    """

    config: KVCacheConfig
    entries: list[KVEntry] = field(default_factory=list)
    _total_tokens_seen: int = 0

    def add_token(self, attention_score: float = 0.0) -> None:
        """Add a new token to the cache."""
        pos = self._total_tokens_seen
        is_sink = pos < self.config.n_sink_tokens
        entry = KVEntry(
            position=pos,
            attention_score=attention_score,
            is_sink=is_sink,
        )
        self.entries.append(entry)
        self._total_tokens_seen += 1

    def evict(self) -> int:
        """Apply the heavy-hitter eviction policy. Returns count of evicted tokens."""
        if len(self.entries) <= self._budget:
            return 0

        cfg = self.config
        kept: list[KVEntry] = []

        # Always keep sink tokens
        sinks = [e for e in self.entries if e.is_sink]

        # Always keep the local window (most recent tokens)
        local_start = max(0, len(self.entries) - cfg.n_local_tokens)
        local = self.entries[local_start:]

        # From the middle, keep top-K by attention score
        sink_positions = {e.position for e in sinks}
        local_positions = {e.position for e in local}
        middle = [
            e for e in self.entries
            if e.position not in sink_positions and e.position not in local_positions
        ]
        middle.sort(key=lambda e: e.attention_score, reverse=True)
        heavy_hitters = middle[:cfg.n_heavy_hitters]

        # Merge and deduplicate
        kept_positions = set()
        for entry_list in [sinks, heavy_hitters, local]:
            for e in entry_list:
                if e.position not in kept_positions:
                    kept.append(e)
                    kept_positions.add(e.position)

        n_evicted = len(self.entries) - len(kept)
        kept.sort(key=lambda e: e.position)
        self.entries = kept
        return n_evicted

    @property
    def _budget(self) -> int:
        cfg = self.config
        return cfg.n_sink_tokens + cfg.n_local_tokens + cfg.n_heavy_hitters

    def quantize_old(self) -> int:
        """Quantize non-local entries to INT4. Returns count of newly quantized."""
        cfg = self.config
        local_start = max(0, len(self.entries) - cfg.n_local_tokens)
        count = 0
        for i, e in enumerate(self.entries):
            if i < local_start and not e.quantized and not e.is_sink:
                e.quantized = True
                count += 1
        return count

    def cache_size_bytes(self) -> int:
        """Current memory footprint of the retained cache."""
        cfg = self.config
        total = 0
        for e in self.entries:
            bpp = cfg.bytes_per_quantized if e.quantized else cfg.bytes_per_fp16
            total += int(cfg.params_per_token * bpp)
        return total

    def cache_size_gb(self) -> float:
        return self.cache_size_bytes() / (1024 ** 3)

    def read_latency_ms(self, hw: HardwareProfile) -> float:
        """RAM read latency to scan the entire cache during attention."""
        gb = self.cache_size_gb()
        return (gb / hw.ram_bandwidth_gb_s) * 1000

    def stats(self) -> dict:
        n_quantized = sum(1 for e in self.entries if e.quantized)
        return {
            "tokens_retained": len(self.entries),
            "tokens_seen": self._total_tokens_seen,
            "tokens_evicted": self._total_tokens_seen - len(self.entries),
            "n_quantized": n_quantized,
            "cache_size_gb": round(self.cache_size_gb(), 4),
        }


def estimate_kv_cache_sizes(
    context_length: int,
    config: KVCacheConfig,
    hw: HardwareProfile,
) -> dict:
    """Compare standard FP16 vs optimized (evicted + quantized) KV cache."""
    # Standard: all tokens in FP16
    std_bytes = context_length * config.params_per_token * config.bytes_per_fp16
    std_gb = std_bytes / (1024 ** 3)
    std_latency = (std_gb / hw.ram_bandwidth_gb_s) * 1000

    # Optimized: evicted + quantized
    retained = min(
        context_length,
        config.n_sink_tokens + config.n_local_tokens + config.n_heavy_hitters,
    )
    # Sink + local in FP16, heavy-hitters in INT4
    n_fp16 = min(context_length, config.n_sink_tokens + config.n_local_tokens)
    n_int4 = max(0, retained - n_fp16)
    opt_bytes = (
        n_fp16 * config.params_per_token * config.bytes_per_fp16
        + n_int4 * config.params_per_token * config.bytes_per_quantized
    )
    opt_gb = opt_bytes / (1024 ** 3)
    opt_latency = (opt_gb / hw.ram_bandwidth_gb_s) * 1000

    return {
        "context_length": context_length,
        "standard": {
            "cache_gb": round(std_gb, 2),
            "latency_ms": round(std_latency, 1),
        },
        "optimized": {
            "tokens_retained": retained,
            "cache_gb": round(opt_gb, 4),
            "latency_ms": round(opt_latency, 1),
        },
        "reduction_x": round(std_latency / opt_latency, 1) if opt_latency > 0 else float("inf"),
    }
