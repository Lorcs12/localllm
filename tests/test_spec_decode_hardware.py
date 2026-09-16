"""Tests for L3 cache, sparse ops, KV cache, and pipeline simulation."""
import numpy as np
import pytest

from neuralbyte.spec_decode.hardware import (
    HardwareProfile,
    L3Cache,
    compute_time_ms,
    memcpy_time_ms,
    simulate_draft_window,
    ssd_read_time_ms,
)
from neuralbyte.spec_decode.sparse_ops import (
    SparseLayerConfig,
    estimate_full_model_sparse_time,
    estimate_sparse_layer_time,
    sparse_matmul_fused,
    sparse_matmul_standard,
    verify_fused_correctness,
)
from neuralbyte.spec_decode.kv_cache import (
    KVCache,
    KVCacheConfig,
    estimate_kv_cache_sizes,
)
from neuralbyte.spec_decode.pipeline import (
    PipelineConfig,
    PipelineSimResult,
    print_pipeline_report,
    simulate_pipeline,
)


# ---------------------------------------------------------------------------
# L3Cache
# ---------------------------------------------------------------------------

class TestL3Cache:
    def test_allocate_fits(self):
        cache = L3Cache(capacity_mb=32.0)
        evicted = cache.allocate("model", 10.0)
        assert evicted is False
        assert cache.has("model")

    def test_allocate_evicts_when_full(self):
        cache = L3Cache(capacity_mb=32.0)
        cache.allocate("model", 30.0)
        evicted = cache.allocate("slabs", 10.0)
        assert evicted is True
        assert not cache.has("model")
        assert cache.has("slabs")

    def test_bypass_skips_cache(self):
        cache = L3Cache(capacity_mb=32.0)
        cache.allocate("model", 30.0)
        evicted = cache.allocate("slabs", 10.0, bypass_l3=True)
        assert evicted is False
        assert cache.has("model")
        assert not cache.has("slabs")

    def test_usage_ratio(self):
        cache = L3Cache(capacity_mb=100.0)
        cache.allocate("a", 25.0)
        assert abs(cache.usage_ratio() - 0.25) < 1e-10


# ---------------------------------------------------------------------------
# simulate_draft_window
# ---------------------------------------------------------------------------

class TestSimulateDraftWindow:
    def test_bypass_passes_pipeline(self):
        hw = HardwareProfile()
        result = simulate_draft_window(hw, draft_tokens=10, bypass_cache=True)
        assert result["pipeline_ok"] is True
        assert result["eviction_occurred"] is False

    def test_no_bypass_may_evict(self):
        hw = HardwareProfile()
        result = simulate_draft_window(
            hw, draft_tokens=10, slab_fetch_mb=40.0, bypass_cache=False
        )
        assert result["eviction_occurred"] is True

    def test_result_keys(self):
        hw = HardwareProfile()
        result = simulate_draft_window(hw, draft_tokens=5, bypass_cache=True)
        assert "total_ms" in result
        assert "tokens" in result
        assert "l3_hits" in result
        assert len(result["tokens"]) == 5


# ---------------------------------------------------------------------------
# Hardware utility functions
# ---------------------------------------------------------------------------

class TestHardwareUtils:
    def test_memcpy_time_positive(self):
        hw = HardwareProfile()
        t = memcpy_time_ms(1_000_000, 2, hw)
        assert t > 0

    def test_compute_time_positive(self):
        hw = HardwareProfile()
        t = compute_time_ms(1_000_000_000, hw)
        assert t > 0

    def test_ssd_sequential_faster_than_random(self):
        hw = HardwareProfile()
        total_bytes = 10 * 1024 * 1024
        seq = ssd_read_time_ms(total_bytes, hw, sequential=True)
        rand = ssd_read_time_ms(total_bytes, hw, sequential=False)
        assert seq < rand


# ---------------------------------------------------------------------------
# Sparse ops
# ---------------------------------------------------------------------------

class TestSparseOps:
    def test_fused_correctness(self):
        result = verify_fused_correctness(d_model=128, ffn_dim=512, n_active=50)
        assert result["match"] is True
        assert result["max_error"] < 1e-10

    def test_standard_vs_fused_same_result(self):
        rng = np.random.RandomState(0)
        W = rng.randn(256, 64).astype(np.float64)
        x = rng.randn(64).astype(np.float64)
        indices = np.array([0, 10, 50, 100, 200], dtype=np.int32)

        standard = sparse_matmul_standard(W, x, indices)
        fused = sparse_matmul_fused(W, x, indices)
        np.testing.assert_allclose(standard, fused, atol=1e-10)

    def test_estimate_fused_faster(self):
        config = SparseLayerConfig()
        hw = HardwareProfile()
        standard = estimate_sparse_layer_time(config, hw, fused=False)
        fused = estimate_sparse_layer_time(config, hw, fused=True)
        assert fused["total_ms"] < standard["total_ms"]

    def test_full_model_estimate(self):
        config = SparseLayerConfig()
        hw = HardwareProfile()
        result = estimate_full_model_sparse_time(config, hw, num_layers=80, fused=True)
        assert result["total_ms"] > 0
        assert result["num_layers"] == 80


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_add_tokens(self):
        config = KVCacheConfig(num_layers=4, kv_heads=2, head_dim=64)
        cache = KVCache(config=config)
        for i in range(10):
            cache.add_token(attention_score=float(i))
        assert len(cache.entries) == 10

    def test_sink_tokens_marked(self):
        config = KVCacheConfig(n_sink_tokens=4)
        cache = KVCache(config=config)
        for i in range(10):
            cache.add_token()
        sinks = [e for e in cache.entries if e.is_sink]
        assert len(sinks) == 4

    def test_eviction_reduces_entries(self):
        config = KVCacheConfig(
            num_layers=4,
            kv_heads=2,
            head_dim=64,
            n_sink_tokens=2,
            n_local_tokens=4,
            n_heavy_hitters=3,
        )
        cache = KVCache(config=config)
        rng = np.random.RandomState(0)
        for i in range(100):
            cache.add_token(attention_score=rng.rand())

        n_before = len(cache.entries)
        n_evicted = cache.evict()
        assert n_evicted > 0
        assert len(cache.entries) < n_before
        assert len(cache.entries) <= 2 + 4 + 3  # sink + local + heavy

    def test_quantize_old(self):
        config = KVCacheConfig(n_sink_tokens=2, n_local_tokens=4)
        cache = KVCache(config=config)
        for i in range(20):
            cache.add_token()
        count = cache.quantize_old()
        assert count > 0

    def test_cache_size_decreases_with_quantization(self):
        config = KVCacheConfig(num_layers=4, kv_heads=2, head_dim=64, n_sink_tokens=2, n_local_tokens=4)
        cache = KVCache(config=config)
        for i in range(20):
            cache.add_token()

        size_before = cache.cache_size_bytes()
        cache.quantize_old()
        size_after = cache.cache_size_bytes()
        assert size_after < size_before

    def test_read_latency_positive(self):
        config = KVCacheConfig(num_layers=4, kv_heads=2, head_dim=64)
        cache = KVCache(config=config)
        for i in range(100):
            cache.add_token()
        hw = HardwareProfile()
        latency = cache.read_latency_ms(hw)
        assert latency > 0

    def test_stats_keys(self):
        config = KVCacheConfig()
        cache = KVCache(config=config)
        cache.add_token()
        s = cache.stats()
        assert "tokens_retained" in s
        assert "cache_size_gb" in s


# ---------------------------------------------------------------------------
# estimate_kv_cache_sizes
# ---------------------------------------------------------------------------

class TestEstimateKVCacheSizes:
    def test_optimized_smaller_than_standard(self):
        config = KVCacheConfig()
        hw = HardwareProfile()
        result = estimate_kv_cache_sizes(32000, config, hw)

        assert result["optimized"]["cache_gb"] < result["standard"]["cache_gb"]
        assert result["optimized"]["latency_ms"] < result["standard"]["latency_ms"]
        assert result["reduction_x"] > 1.0


# ---------------------------------------------------------------------------
# Pipeline simulation
# ---------------------------------------------------------------------------

class TestPipelineSimulation:
    def test_simulate_default(self):
        result = simulate_pipeline()
        assert isinstance(result, PipelineSimResult)
        assert result.total_tokens >= 1000
        assert result.tokens_per_second > 0
        assert result.total_rounds > 0

    def test_result_has_phase_totals(self):
        result = simulate_pipeline()
        assert "predict" in result.phase_totals
        assert "draft_fetch" in result.phase_totals
        assert "verify" in result.phase_totals
        assert "smw_update" in result.phase_totals

    def test_bottleneck_is_valid_phase(self):
        result = simulate_pipeline()
        assert result.bottleneck in {"predict", "draft_fetch", "verify", "smw_update"}

    def test_custom_config(self):
        config = PipelineConfig(
            draft_tokens_per_round=5,
            acceptance_rate=0.80,
            total_tokens=100,
        )
        result = simulate_pipeline(config)
        assert result.total_tokens >= 100
        assert result.acceptance_rate == 0.80

    def test_print_report(self):
        result = simulate_pipeline()
        report = print_pipeline_report(result)
        assert "ADAPTIVE-EAGLE" in report
        assert "tokens/sec" in report
        assert "KV Cache" in report

    def test_cache_bypass_stats(self):
        result = simulate_pipeline()
        cb = result.cache_bypass_stats
        assert "without_bypass_ms" in cb
        assert "with_bypass_ms" in cb
        assert cb["bypass_saves_ms"] >= 0
