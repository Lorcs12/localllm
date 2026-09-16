"""Tests for neuralbyte.spec_decode.hybrid_mamba — architecture comparison simulation."""
import pytest

from neuralbyte.spec_decode.hardware import HardwareProfile
from neuralbyte.spec_decode.hybrid_mamba import (
    ArchitectureComparison,
    HybridMambaConfig,
    ModelSpec,
    TransformerConfig,
    compare_architectures,
    print_architecture_report,
    simulate_generation_loop,
    simulate_hybrid_mamba,
    simulate_transformer,
)


# ── ModelSpec ──────────────────────────────────────────────────

class TestModelSpec:
    def test_defaults(self):
        spec = ModelSpec()
        assert spec.params == 70_000_000_000
        assert spec.layers == 80
        assert spec.kv_heads == 8
        assert spec.head_dim == 128
        assert spec.bytes_per_param == 2

    def test_kv_entry_bytes(self):
        spec = ModelSpec()
        # 2 (K+V) × 128 (head_dim) × 2 (FP16) = 512 bytes
        assert spec.kv_entry_bytes == 512

    def test_mamba_state_per_layer(self):
        spec = ModelSpec()
        # 5120 × 128 × 2 = 1,310,720
        assert spec.mamba_state_per_layer_bytes == 5120 * 128 * 2


# ── TransformerConfig ─────────────────────────────────────────

class TestTransformerConfig:
    def test_kv_bytes_per_token(self):
        cfg = TransformerConfig()
        # 80 layers × 8 heads × 512 bytes = 327,680
        assert cfg.kv_bytes_per_token == 80 * 8 * 512

    def test_kv_cache_at_1m_exceeds_64gb(self):
        cfg = TransformerConfig()
        kv_gb = cfg.kv_cache_gb(1_000_000)
        assert kv_gb > 64.0

    def test_kv_cache_at_1m_approximately_305gb(self):
        cfg = TransformerConfig()
        kv_gb = cfg.kv_cache_gb(1_000_000)
        assert 300.0 < kv_gb < 310.0


# ── HybridMambaConfig ────────────────────────────────────────

class TestHybridMambaConfig:
    def test_defaults(self):
        cfg = HybridMambaConfig()
        assert cfg.attn_layers == 10
        assert cfg.mamba_layers == 70
        assert cfg.heavy_hitter_tokens == 2048

    def test_kv_bytes_per_token_only_attn_layers(self):
        cfg = HybridMambaConfig()
        full = TransformerConfig()
        # 10/80 = 1/8 of the full transformer KV per token
        assert cfg.kv_bytes_per_token == full.kv_bytes_per_token * 10 // 80

    def test_mamba_state_fixed_size(self):
        cfg = HybridMambaConfig()
        # 70 layers × 5120 × 128 × 2 bytes
        expected = 70 * 5120 * 128 * 2
        assert cfg.mamba_state_bytes == expected

    def test_mamba_state_mb_approximately_89(self):
        cfg = HybridMambaConfig()
        assert 85.0 < cfg.mamba_state_mb < 95.0

    def test_hot_kv_mb_approximately_80(self):
        cfg = HybridMambaConfig()
        assert 75.0 < cfg.hot_kv_mb() < 85.0

    def test_total_ram_under_200mb(self):
        cfg = HybridMambaConfig()
        assert cfg.total_ram_mb() < 200.0

    def test_total_ram_is_mamba_plus_hot_kv(self):
        cfg = HybridMambaConfig()
        assert cfg.total_ram_bytes() == cfg.mamba_state_bytes + cfg.hot_kv_bytes()


# ── simulate_transformer ─────────────────────────────────────

class TestSimulateTransformer:
    def test_returns_expected_keys(self):
        result = simulate_transformer()
        for key in ("architecture", "kv_cache_gb", "fits_in_ram", "ttft_seconds", "ttft_minutes"):
            assert key in result

    def test_architecture_label(self):
        result = simulate_transformer()
        assert result["architecture"] == "standard_transformer"

    def test_does_not_fit_in_64gb_at_1m(self):
        result = simulate_transformer(context_len=1_000_000, ram_capacity_gb=64.0)
        assert result["fits_in_ram"] is False

    def test_ttft_over_1000_seconds_at_1m(self):
        result = simulate_transformer(context_len=1_000_000)
        assert result["ttft_seconds"] > 1000.0

    def test_small_context_fits_in_ram(self):
        result = simulate_transformer(context_len=1000, ram_capacity_gb=64.0)
        assert result["fits_in_ram"] is True


# ── simulate_hybrid_mamba ─────────────────────────────────────

class TestSimulateHybridMamba:
    def test_returns_expected_keys(self):
        result = simulate_hybrid_mamba()
        for key in ("architecture", "mamba_state_mb", "hot_kv_mb", "total_ram_mb", "ttft_ms"):
            assert key in result

    def test_architecture_label(self):
        result = simulate_hybrid_mamba()
        assert result["architecture"] == "hybrid_mamba"

    def test_total_ram_under_200mb(self):
        result = simulate_hybrid_mamba()
        assert result["total_ram_mb"] < 200.0

    def test_ttft_under_100ms(self):
        result = simulate_hybrid_mamba()
        assert result["ttft_ms"] < 100.0

    def test_cold_kv_fetch_under_1ms(self):
        result = simulate_hybrid_mamba()
        assert result["cold_kv_fetch_ms"] < 1.0


# ── compare_architectures ────────────────────────────────────

class TestCompareArchitectures:
    def test_returns_comparison_object(self):
        cmp = compare_architectures()
        assert isinstance(cmp, ArchitectureComparison)

    def test_memory_reduction_over_1000x(self):
        cmp = compare_architectures(context_len=1_000_000)
        assert cmp.memory_reduction_x > 1000.0

    def test_ttft_reduction_over_100000x(self):
        cmp = compare_architectures(context_len=1_000_000)
        assert cmp.ttft_reduction_x > 100_000

    def test_transformer_fatal_at_64gb(self):
        cmp = compare_architectures(context_len=1_000_000, ram_capacity_gb=64.0)
        assert cmp.transformer_fatal is True

    def test_transformer_not_fatal_at_512gb(self):
        cmp = compare_architectures(context_len=1_000_000, ram_capacity_gb=512.0)
        assert cmp.transformer_fatal is False


# ── simulate_generation_loop ─────────────────────────────────

class TestSimulateGenerationLoop:
    def test_thread_b_hidden_in_draft_window(self):
        result = simulate_generation_loop()
        assert result["fetch_hidden"] is True

    def test_thread_b_under_1ms(self):
        result = simulate_generation_loop()
        assert result["thread_b_ms"] < 1.0

    def test_zero_overhead_when_hidden(self):
        result = simulate_generation_loop()
        assert result["effective_overhead_ms"] == 0.0

    def test_verdict_when_hidden(self):
        result = simulate_generation_loop()
        assert "infinite context" in result["verdict"].lower()

    def test_slow_ssd_not_hidden(self):
        hw = HardwareProfile(ssd_sequential_gb_s=0.01)
        result = simulate_generation_loop(hw=hw, draft_window_ms=1.0)
        assert result["fetch_hidden"] is False
        assert result["effective_overhead_ms"] > 0.0


# ── print_architecture_report ─────────────────────────────────

class TestPrintArchitectureReport:
    def test_report_contains_sections(self):
        report = print_architecture_report()
        assert "Standard Transformer" in report
        assert "Hybrid Mamba" in report
        assert "REDUCTION" in report

    def test_report_shows_fatal_for_64gb(self):
        cmp = compare_architectures(context_len=1_000_000, ram_capacity_gb=64.0)
        report = print_architecture_report(cmp)
        assert "FATAL" in report

    def test_report_contains_generation_loop(self):
        report = print_architecture_report()
        assert "Thread A" in report
        assert "Thread B" in report
