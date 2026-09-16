"""Tests for neuralbyte.spec_decode.hybrid_engine — hybrid CPU+GPU inference simulation."""
import pytest

from neuralbyte.spec_decode.hardware import HardwareProfile
from neuralbyte.spec_decode.inference_engine import ModelConfig
from neuralbyte.spec_decode.hybrid_engine import (
    HybridEngineConfig,
    compare_hybrid_vs_tiers,
    compare_optimizations,
    print_hybrid_engine_report,
    print_optimization_report,
    simulate_hybrid_engine,
)


# ---------------------------------------------------------------------------
# HybridEngineConfig
# ---------------------------------------------------------------------------

class TestHybridEngineConfig:

    def test_defaults(self):
        cfg = HybridEngineConfig()
        assert cfg.d_model == 8192
        assert cfg.ffn_dim == 28672
        assert cfg.attention_ratio == 0.35
        assert cfg.ffn_ratio == 0.65
        assert cfg.sparsity == 0.90
        assert cfg.hot_neuron_ratio == 0.10
        assert cfg.avx512_speedup == 2.0
        assert cfg.gdeflate_ratio == 1.5
        assert cfg.vram_reserve_gb == 0.5
        assert cfg.sync_vector_bytes == 65536

    def test_params_per_layer(self):
        cfg = HybridEngineConfig()
        assert cfg.params_per_layer == 70_000_000_000 // 80

    def test_dense_per_layer_bytes(self):
        cfg = HybridEngineConfig()
        expected = int(cfg.dense_per_layer_params * 0.5)
        assert cfg.dense_per_layer_bytes == expected
        assert 140_000_000 < cfg.dense_per_layer_bytes < 170_000_000

    def test_sparse_per_layer_bytes(self):
        cfg = HybridEngineConfig()
        expected = int(cfg.ffn_per_layer_params * 0.5)
        assert cfg.sparse_per_layer_bytes == expected
        assert 260_000_000 < cfg.sparse_per_layer_bytes < 300_000_000

    def test_hot_neuron_bytes(self):
        cfg = HybridEngineConfig()
        expected = int(int(cfg.ffn_per_layer_params * 0.10) * 0.5)
        assert cfg.hot_neuron_bytes == expected
        assert 25_000_000 < cfg.hot_neuron_bytes < 35_000_000

    def test_cold_neuron_bytes(self):
        cfg = HybridEngineConfig()
        assert cfg.cold_neuron_bytes == int(cfg.cold_neuron_params * 0.5)
        assert cfg.cold_neuron_bytes > cfg.hot_neuron_bytes

    def test_hot_vram_gb_fits_in_4gb(self):
        cfg = HybridEngineConfig()
        assert cfg.hot_vram_gb + cfg.vram_reserve_gb < 4.0

    def test_active_cold_neurons(self):
        cfg = HybridEngineConfig()
        expected = int(cfg.cold_neuron_params * (1 - 0.90))
        assert cfg.active_cold_neurons == expected

    def test_active_cold_flops(self):
        cfg = HybridEngineConfig()
        assert cfg.active_cold_flops == 2 * cfg.active_cold_neurons

    def test_dense_plus_hot_flops(self):
        cfg = HybridEngineConfig()
        total = cfg.dense_flops_per_layer + cfg.hot_flops_per_layer
        assert 700_000_000 < total < 750_000_000


# ---------------------------------------------------------------------------
# simulate_hybrid_engine
# ---------------------------------------------------------------------------

class TestSimulateHybridEngine:

    def test_returns_expected_keys(self):
        r = simulate_hybrid_engine()
        expected_keys = {
            "feasible", "fatal_reason", "gpu_io_ms", "gpu_compute_ms",
            "gpu_steady_ms", "cpu_sparse_ram_ms", "cpu_sparse_ssd_ms",
            "sync_ms", "per_layer_wall_ms", "forward_pass_ms", "ttft_ms",
            "tokens_per_second", "gpu_utilization", "cpu_utilization",
            "hot_vram_mb", "hot_vram_gb", "cold_ram_mb", "cold_ram_gb",
            "cold_layers_in_ram", "cold_layers_overflow", "transfer_ms",
            "decompress_ms", "bottleneck", "per_layer",
        }
        assert expected_keys.issubset(set(r.keys()))

    def test_feasible_with_default_config(self):
        r = simulate_hybrid_engine()
        assert r["feasible"] is True
        assert r["fatal_reason"] is None

    def test_gpu_io_dominates(self):
        r = simulate_hybrid_engine()
        assert r["gpu_io_ms"] > r["gpu_compute_ms"] * 10

    def test_bottleneck_is_gpu_io(self):
        r = simulate_hybrid_engine()
        assert r["bottleneck"] == "gpu_io"

    def test_forward_pass_between_2000_and_3000(self):
        r = simulate_hybrid_engine()
        assert 2000 < r["forward_pass_ms"] < 3000

    def test_tokens_per_second_between_0_3_and_0_6(self):
        r = simulate_hybrid_engine()
        assert 0.3 < r["tokens_per_second"] < 0.6

    def test_sync_negligible(self):
        r = simulate_hybrid_engine()
        assert r["sync_ms"] < 0.1

    def test_gpu_utilization_very_low(self):
        r = simulate_hybrid_engine()
        assert r["gpu_utilization"] < 0.05

    def test_per_layer_has_correct_count(self):
        r = simulate_hybrid_engine()
        assert len(r["per_layer"]) == 80

    def test_infeasible_when_vram_too_small(self):
        hw = HardwareProfile(gpu_vram_gb=1.0, ssd_sequential_gb_s=3.5, pcie_bandwidth_gb_s=12.0)
        r = simulate_hybrid_engine(hw=hw)
        assert r["feasible"] is False
        assert "VRAM" in r["fatal_reason"]

    def test_cold_layers_overflow(self):
        r = simulate_hybrid_engine()
        assert r["cold_layers_in_ram"] < 80
        assert r["cold_layers_overflow"] > 0
        assert r["cold_layers_in_ram"] + r["cold_layers_overflow"] == 80

    def test_per_layer_cold_source(self):
        r = simulate_hybrid_engine()
        ram_layers = [p for p in r["per_layer"] if p["cold_source"] == "ram"]
        ssd_layers = [p for p in r["per_layer"] if p["cold_source"] == "ssd"]
        assert len(ram_layers) == r["cold_layers_in_ram"]
        assert len(ssd_layers) == r["cold_layers_overflow"]


# ---------------------------------------------------------------------------
# simulate_hybrid_engine — custom configs
# ---------------------------------------------------------------------------

class TestSimulateHybridEngineCustom:

    def test_high_sparsity_reduces_cpu_time(self):
        cfg_high = HybridEngineConfig(sparsity=0.95)
        cfg_low = HybridEngineConfig(sparsity=0.80)
        r_high = simulate_hybrid_engine(cfg_high)
        r_low = simulate_hybrid_engine(cfg_low)
        assert r_high["cpu_sparse_ram_ms"] < r_low["cpu_sparse_ram_ms"]

    def test_higher_hot_ratio_increases_vram(self):
        cfg_lo = HybridEngineConfig(hot_neuron_ratio=0.05)
        cfg_hi = HybridEngineConfig(hot_neuron_ratio=0.20)
        r_lo = simulate_hybrid_engine(cfg_lo)
        r_hi = simulate_hybrid_engine(cfg_hi)
        assert r_hi["hot_vram_gb"] > r_lo["hot_vram_gb"]

    def test_fast_ssd_reduces_gpu_io(self):
        hw_slow = HardwareProfile(ssd_sequential_gb_s=2.0, pcie_bandwidth_gb_s=12.0)
        hw_fast = HardwareProfile(ssd_sequential_gb_s=7.0, pcie_bandwidth_gb_s=12.0)
        r_slow = simulate_hybrid_engine(hw=hw_slow)
        r_fast = simulate_hybrid_engine(hw=hw_fast)
        assert r_fast["gpu_io_ms"] < r_slow["gpu_io_ms"]

    def test_no_sparsity_maximizes_cpu_load(self):
        cfg = HybridEngineConfig(sparsity=0.0)
        r = simulate_hybrid_engine(cfg)
        assert r["cpu_sparse_ram_ms"] > 0

    def test_all_cold_in_ram_with_large_ram(self):
        hw = HardwareProfile(ram_capacity_gb=64.0, ssd_sequential_gb_s=3.5, pcie_bandwidth_gb_s=12.0)
        r = simulate_hybrid_engine(hw=hw)
        assert r["cold_layers_overflow"] == 0
        assert r["cold_layers_in_ram"] == 80


# ---------------------------------------------------------------------------
# compare_hybrid_vs_tiers
# ---------------------------------------------------------------------------

class TestCompareHybridVsTiers:

    def test_returns_expected_keys(self):
        c = compare_hybrid_vs_tiers()
        assert "hybrid" in c
        assert "ggml_zigzag" in c
        assert "directstorage" in c
        assert "speedup_vs_zigzag" in c
        assert "speedup_vs_directstorage" in c

    def test_hybrid_faster_than_directstorage(self):
        c = compare_hybrid_vs_tiers()
        assert c["speedup_vs_directstorage"] > 1.0

    def test_hybrid_faster_than_zigzag(self):
        c = compare_hybrid_vs_tiers()
        assert c["speedup_vs_zigzag"] > 1.0

    def test_speedup_vs_directstorage_range(self):
        c = compare_hybrid_vs_tiers()
        assert 1.5 < c["speedup_vs_directstorage"] < 5.0

    def test_all_tiers_have_tokens_per_second(self):
        c = compare_hybrid_vs_tiers()
        assert c["hybrid"]["tokens_per_second"] > 0
        assert c["ggml_zigzag"]["tokens_per_second"] > 0
        assert c["directstorage"]["tokens_per_second"] > 0


# ---------------------------------------------------------------------------
# print_hybrid_engine_report
# ---------------------------------------------------------------------------

class TestPrintHybridEngineReport:

    def test_report_contains_sections(self):
        report = print_hybrid_engine_report()
        assert "GPU Dense Path" in report
        assert "CPU Sparse Path" in report
        assert "Sync" in report
        assert "Throughput" in report

    def test_report_contains_architecture_diagram(self):
        report = print_hybrid_engine_report()
        assert "SSD" in report
        assert "AVX-512" in report
        assert "GDeflate" in report

    def test_report_shows_bottleneck(self):
        report = print_hybrid_engine_report()
        assert "gpu_io" in report


# ---------------------------------------------------------------------------
# I/O Optimizations
# ---------------------------------------------------------------------------

class TestOptimizations:

    def test_temporal_cache_reduces_gpu_io(self):
        cfg = HybridEngineConfig(temporal_delta_ratio=0.03)
        r = simulate_hybrid_engine(cfg)
        assert r["gpu_io_ms"] < 2.0

    def test_speculative_increases_throughput(self):
        cfg = HybridEngineConfig(speculative_tokens=7)
        r = simulate_hybrid_engine(cfg)
        assert r["tokens_per_second"] > 1.0

    def test_ml_cache_reduces_cold_load(self):
        cfg_base = HybridEngineConfig()
        cfg_ml = HybridEngineConfig(ml_cache_speedup=2.6)
        r_base = simulate_hybrid_engine(cfg_base)
        r_ml = simulate_hybrid_engine(cfg_ml)
        assert r_ml["cpu_sparse_ssd_ms"] < r_base["cpu_sparse_ssd_ms"]

    def test_combined_throughput_over_20(self):
        cfg = HybridEngineConfig(
            temporal_delta_ratio=0.03,
            speculative_tokens=7,
            ml_cache_speedup=2.6,
        )
        r = simulate_hybrid_engine(cfg)
        assert r["tokens_per_second"] > 20.0

    def test_defaults_match_base(self):
        r_default = simulate_hybrid_engine()
        r_explicit = simulate_hybrid_engine(HybridEngineConfig(
            temporal_delta_ratio=1.0,
            speculative_tokens=1,
            ml_cache_speedup=1.0,
        ))
        assert r_default["tokens_per_second"] == r_explicit["tokens_per_second"]
        assert r_default["gpu_io_ms"] == r_explicit["gpu_io_ms"]

    def test_temporal_narrows_bottleneck_gap(self):
        r_base = simulate_hybrid_engine()
        cfg = HybridEngineConfig(temporal_delta_ratio=0.03)
        r_opt = simulate_hybrid_engine(cfg)
        base_ratio = r_base["gpu_compute_ms"] / r_base["gpu_io_ms"]
        opt_ratio = r_opt["gpu_compute_ms"] / r_opt["gpu_io_ms"]
        assert opt_ratio > base_ratio * 10

    def test_speculative_round_ms_includes_draft_overhead(self):
        cfg = HybridEngineConfig(speculative_tokens=7)
        r = simulate_hybrid_engine(cfg)
        assert r["round_ms"] > r["forward_pass_ms"]
        assert r["round_ms"] == pytest.approx(r["forward_pass_ms"] + 80.0, abs=0.1)

    def test_result_contains_optimization_keys(self):
        r = simulate_hybrid_engine()
        assert "temporal_delta_ratio" in r
        assert "speculative_tokens" in r
        assert "ml_cache_speedup" in r
        assert "tokens_per_round" in r
        assert "round_ms" in r


# ---------------------------------------------------------------------------
# compare_optimizations
# ---------------------------------------------------------------------------

class TestCompareOptimizations:

    def test_returns_five_configs(self):
        c = compare_optimizations()
        assert "base" in c
        assert "temporal" in c
        assert "speculative" in c
        assert "ml_cache" in c
        assert "combined" in c

    def test_combined_fastest(self):
        c = compare_optimizations()
        combined_tps = c["combined"]["tokens_per_second"]
        for key in ("base", "temporal", "speculative", "ml_cache"):
            assert combined_tps > c[key]["tokens_per_second"]

    def test_temporal_bigger_impact_than_ml_cache(self):
        c = compare_optimizations()
        assert c["temporal"]["speedup_vs_base"] > c["ml_cache"]["speedup_vs_base"]

    def test_all_feasible(self):
        c = compare_optimizations()
        for key in ("base", "temporal", "speculative", "ml_cache", "combined"):
            assert c[key]["feasible"] is True

    def test_speedup_ratios_positive(self):
        c = compare_optimizations()
        for key in ("base", "temporal", "speculative", "ml_cache", "combined"):
            assert c[key]["speedup_vs_base"] >= 1.0


# ---------------------------------------------------------------------------
# print_optimization_report
# ---------------------------------------------------------------------------

class TestPrintOptimizationReport:

    def test_report_contains_all_configs(self):
        report = print_optimization_report()
        assert "Base" in report
        assert "Temporal" in report
        assert "Speculative" in report
        assert "ML Cache" in report
        assert "Combined" in report

    def test_report_contains_analysis(self):
        report = print_optimization_report()
        assert "Analysis" in report

    def test_report_contains_paper_references(self):
        report = print_optimization_report()
        assert "LLM in a Flash" in report
        assert "FlashMoE" in report
