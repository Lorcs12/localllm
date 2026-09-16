"""Tests for neuralbyte.spec_decode.inference_engine — three-tier inference comparison."""
import pytest

from neuralbyte.spec_decode.hardware import HardwareProfile
from neuralbyte.spec_decode.inference_engine import (
    DirectStorageConfig,
    InferenceTier,
    ModelConfig,
    PyTorchConfig,
    TierResult,
    ZigZagConfig,
    compare_tiers,
    print_inference_engine_report,
    simulate_directstorage,
    simulate_ggml_zigzag,
    simulate_pytorch,
)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:

    def test_defaults(self):
        m = ModelConfig()
        assert m.params == 70_000_000_000
        assert m.layers == 80
        assert m.bytes_per_param_fp16 == 2
        assert m.bytes_per_param_int4 == 0.5

    def test_fp16_model_gb(self):
        m = ModelConfig()
        assert 120 < m.fp16_model_gb < 140

    def test_int4_model_gb(self):
        m = ModelConfig()
        assert 30 < m.int4_model_gb < 35

    def test_quantization_ratio(self):
        m = ModelConfig()
        assert m.quantization_ratio == 4.0

    def test_per_layer_bytes(self):
        m = ModelConfig()
        assert m.fp16_per_layer_bytes == m.fp16_model_bytes // 80
        assert m.int4_per_layer_bytes == m.int4_model_bytes // 80


# ---------------------------------------------------------------------------
# Tier 1: PyTorch
# ---------------------------------------------------------------------------

class TestSimulatePytorch:

    def test_infeasible_on_consumer_hardware(self):
        hw = HardwareProfile(gpu_vram_gb=4.0, ram_capacity_gb=16.0)
        r = simulate_pytorch(hw=hw)
        assert r.feasible is False
        assert r.tier == InferenceTier.PYTORCH

    def test_fatal_reason_mentions_model_size(self):
        hw = HardwareProfile(gpu_vram_gb=4.0, ram_capacity_gb=16.0)
        r = simulate_pytorch(hw=hw)
        assert "exceeds" in r.fatal_reason

    def test_zero_throughput_when_infeasible(self):
        hw = HardwareProfile(gpu_vram_gb=4.0, ram_capacity_gb=16.0)
        r = simulate_pytorch(hw=hw)
        assert r.tokens_per_second == 0.0

    def test_feasible_with_huge_vram(self):
        hw = HardwareProfile(gpu_vram_gb=256.0, ram_capacity_gb=512.0)
        r = simulate_pytorch(hw=hw)
        assert r.feasible is True
        assert r.tokens_per_second > 0

    def test_pcie_crossings_is_two(self):
        r = simulate_pytorch()
        assert r.pcie_crossings == 2


# ---------------------------------------------------------------------------
# Tier 2: GGML + ZigZag
# ---------------------------------------------------------------------------

class TestSimulateGgmlZigzag:

    def test_feasible_on_consumer_hardware(self):
        r = simulate_ggml_zigzag()
        assert r.feasible is True
        assert r.tier == InferenceTier.GGML_ZIGZAG

    def test_ram_bounded_by_buffer(self):
        r = simulate_ggml_zigzag()
        assert r.ram_needed_gb < 4.0

    def test_io_bound(self):
        r = simulate_ggml_zigzag()
        assert r.bottleneck != "compute"

    def test_gpu_utilization_low(self):
        r = simulate_ggml_zigzag()
        assert r.gpu_utilization < 0.05

    def test_throughput_positive(self):
        r = simulate_ggml_zigzag()
        assert r.tokens_per_second > 0

    def test_per_layer_populated(self):
        r = simulate_ggml_zigzag()
        assert len(r.per_layer) == 80

    def test_pcie_crossings_is_two(self):
        r = simulate_ggml_zigzag()
        assert r.pcie_crossings == 2


# ---------------------------------------------------------------------------
# Tier 3: DirectStorage
# ---------------------------------------------------------------------------

class TestSimulateDirectstorage:

    def test_feasible_on_consumer_hardware(self):
        r = simulate_directstorage()
        assert r.feasible is True
        assert r.tier == InferenceTier.DIRECTSTORAGE

    def test_near_zero_ram(self):
        r = simulate_directstorage()
        assert r.ram_needed_gb < 0.1

    def test_pcie_crossings_is_one(self):
        r = simulate_directstorage()
        assert r.pcie_crossings == 1

    def test_faster_than_zigzag(self):
        zz = simulate_ggml_zigzag()
        ds = simulate_directstorage()
        assert ds.tokens_per_second > zz.tokens_per_second

    def test_throughput_positive(self):
        r = simulate_directstorage()
        assert r.tokens_per_second > 0

    def test_per_layer_populated(self):
        r = simulate_directstorage()
        assert len(r.per_layer) == 80


# ---------------------------------------------------------------------------
# compare_tiers
# ---------------------------------------------------------------------------

class TestCompareTiers:

    def test_returns_expected_keys(self):
        c = compare_tiers()
        assert "model" in c
        assert "hardware" in c
        assert "tiers" in c
        assert "winner" in c
        assert "speedup_ds_vs_zigzag" in c

    def test_winner_is_directstorage(self):
        c = compare_tiers()
        assert c["winner"] == "directstorage"

    def test_speedup_positive(self):
        c = compare_tiers()
        assert c["speedup_ds_vs_zigzag"] > 1.0

    def test_pytorch_infeasible(self):
        c = compare_tiers()
        assert c["tiers"]["pytorch"]["feasible"] is False

    def test_all_tiers_present(self):
        c = compare_tiers()
        assert "pytorch" in c["tiers"]
        assert "ggml_zigzag" in c["tiers"]
        assert "directstorage" in c["tiers"]


# ---------------------------------------------------------------------------
# print_inference_engine_report
# ---------------------------------------------------------------------------

class TestPrintInferenceEngineReport:

    def test_report_contains_all_tiers(self):
        report = print_inference_engine_report()
        assert "TIER 1" in report
        assert "TIER 2" in report
        assert "TIER 3" in report

    def test_report_contains_comparison(self):
        report = print_inference_engine_report()
        assert "COMPARISON" in report

    def test_report_shows_fatal_for_pytorch(self):
        report = print_inference_engine_report()
        assert "FATAL" in report

    def test_report_shows_winner(self):
        report = print_inference_engine_report()
        assert "WINNER" in report
