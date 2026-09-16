"""Tests for layer-streaming inference: memory estimation, timing simulation,
and functional forward pass with all three streaming modes."""
import numpy as np
import pytest

from neuralbyte.spec_decode.hardware import HardwareProfile
from neuralbyte.spec_decode.layer_stream import (
    LayerBuffer,
    LayerStreamConfig,
    LayerStreamResult,
    LayerStreamRunner,
    StreamMode,
    compare_modes,
    estimate_memory,
    print_layer_stream_report,
    simulate_layer_stream,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return LayerStreamConfig(
        num_layers=4,
        layer_size_bytes=1 * 1024 * 1024,  # 1 MiB per layer (small for fast tests)
        compute_flops_per_layer=100_000_000,  # 100 MFLOPs
        dtype_bytes=4,
    )


@pytest.fixture
def hw():
    return HardwareProfile()


# ---------------------------------------------------------------------------
# estimate_memory
# ---------------------------------------------------------------------------

class TestEstimateMemory:
    def test_static_uses_full_model(self, config):
        mem = estimate_memory(config, StreamMode.STATIC)
        assert mem["buffer_bytes"] == config.total_model_bytes
        assert mem["reduction_x"] == 1.0

    def test_stream_1buf_uses_one_layer(self, config):
        mem = estimate_memory(config, StreamMode.STREAM_1BUF)
        assert mem["buffer_bytes"] == config.layer_size_bytes
        assert mem["reduction_x"] == config.num_layers

    def test_pingpong_uses_two_layers(self, config):
        mem = estimate_memory(config, StreamMode.PINGPONG_2BUF)
        assert mem["buffer_bytes"] == 2 * config.layer_size_bytes
        assert mem["reduction_x"] == config.num_layers / 2

    def test_memory_ordering(self, config):
        static = estimate_memory(config, StreamMode.STATIC)
        pingpong = estimate_memory(config, StreamMode.PINGPONG_2BUF)
        stream = estimate_memory(config, StreamMode.STREAM_1BUF)

        assert stream["buffer_bytes"] < pingpong["buffer_bytes"] < static["buffer_bytes"]

    def test_result_keys(self, config):
        mem = estimate_memory(config, StreamMode.STATIC)
        assert "buffer_bytes" in mem
        assert "buffer_mb" in mem
        assert "static_bytes" in mem
        assert "static_mb" in mem
        assert "reduction_x" in mem
        assert "mode" in mem


# ---------------------------------------------------------------------------
# simulate_layer_stream
# ---------------------------------------------------------------------------

class TestSimulateLayerStream:
    def test_static_no_overlap(self, config, hw):
        r = simulate_layer_stream(config, hw, StreamMode.STATIC)
        assert r.mode == StreamMode.STATIC
        assert r.overlap_saved_ms == 0.0
        assert r.total_time_ms > 0
        assert r.memory_reduction_vs_static == 1.0

    def test_stream_1buf_no_overlap(self, config, hw):
        r = simulate_layer_stream(config, hw, StreamMode.STREAM_1BUF)
        assert r.mode == StreamMode.STREAM_1BUF
        assert r.overlap_saved_ms == 0.0
        assert r.peak_memory_bytes == config.layer_size_bytes

    def test_pingpong_has_overlap(self, config, hw):
        r = simulate_layer_stream(config, hw, StreamMode.PINGPONG_2BUF)
        assert r.mode == StreamMode.PINGPONG_2BUF
        assert r.overlap_saved_ms >= 0.0
        assert r.peak_memory_bytes == 2 * config.layer_size_bytes

    def test_pingpong_faster_or_equal_to_1buf(self, config, hw):
        stream = simulate_layer_stream(config, hw, StreamMode.STREAM_1BUF)
        pingpong = simulate_layer_stream(config, hw, StreamMode.PINGPONG_2BUF)
        assert pingpong.total_time_ms <= stream.total_time_ms + 0.001

    def test_memory_reduction_ratios(self, config, hw):
        static = simulate_layer_stream(config, hw, StreamMode.STATIC)
        stream = simulate_layer_stream(config, hw, StreamMode.STREAM_1BUF)
        pingpong = simulate_layer_stream(config, hw, StreamMode.PINGPONG_2BUF)

        assert stream.memory_reduction_vs_static > pingpong.memory_reduction_vs_static
        assert pingpong.memory_reduction_vs_static > static.memory_reduction_vs_static

    def test_per_layer_populated(self, config, hw):
        r = simulate_layer_stream(config, hw, StreamMode.STREAM_1BUF)
        assert len(r.per_layer) == config.num_layers
        for entry in r.per_layer:
            assert "layer" in entry
            assert "compute_ms" in entry

    def test_io_and_compute_positive(self, config, hw):
        for mode in StreamMode:
            r = simulate_layer_stream(config, hw, mode)
            assert r.io_time_ms > 0
            assert r.compute_time_ms > 0

    def test_total_equals_io_plus_compute_minus_overlap(self, config, hw):
        for mode in StreamMode:
            r = simulate_layer_stream(config, hw, mode)
            expected = r.io_time_ms + r.compute_time_ms - r.overlap_saved_ms
            assert abs(r.total_time_ms - expected) < 0.01


# ---------------------------------------------------------------------------
# compare_modes
# ---------------------------------------------------------------------------

class TestCompareModes:
    def test_returns_all_modes(self, config, hw):
        result = compare_modes(config, hw)
        assert "static" in result["modes"]
        assert "stream_1buf" in result["modes"]
        assert "pingpong_2buf" in result["modes"]

    def test_config_info(self, config, hw):
        result = compare_modes(config, hw)
        assert result["config"]["num_layers"] == config.num_layers
        assert result["config"]["layer_size_mb"] == config.layer_size_mb

    def test_memory_ordering_in_comparison(self, config, hw):
        result = compare_modes(config, hw)
        modes = result["modes"]
        assert modes["stream_1buf"]["peak_memory_mb"] < modes["pingpong_2buf"]["peak_memory_mb"]
        assert modes["pingpong_2buf"]["peak_memory_mb"] < modes["static"]["peak_memory_mb"]

    def test_default_args(self):
        result = compare_modes()
        assert "modes" in result
        assert len(result["modes"]) == 3


# ---------------------------------------------------------------------------
# print_layer_stream_report
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_report_contains_sections(self, config, hw):
        comparison = compare_modes(config, hw)
        report = print_layer_stream_report(comparison)
        assert "LAYER-STREAMING" in report
        assert "Model:" in report
        assert "Key Results:" in report
        assert "RAM reduction" in report

    def test_default_report(self):
        report = print_layer_stream_report()
        assert "LAYER-STREAMING" in report


# ---------------------------------------------------------------------------
# LayerBuffer
# ---------------------------------------------------------------------------

class TestLayerBuffer:
    def test_initial_state(self):
        buf = LayerBuffer()
        assert not buf.is_loaded()
        assert buf.layer_id == -1

    def test_clear(self):
        buf = LayerBuffer(data=np.zeros(10), layer_id=5, size_bytes=40)
        assert buf.is_loaded()
        buf.clear()
        assert not buf.is_loaded()
        assert buf.layer_id == -1


# ---------------------------------------------------------------------------
# LayerStreamRunner
# ---------------------------------------------------------------------------

class TestLayerStreamRunner:
    @pytest.fixture
    def small_config(self):
        return LayerStreamConfig(
            num_layers=3,
            layer_size_bytes=256 * 4,  # 256 float32 = 1 KiB -> 16x16 matrix
            compute_flops_per_layer=1_000_000,
            dtype_bytes=4,
        )

    def test_static_forward(self, small_config, hw):
        runner = LayerStreamRunner(config=small_config, mode=StreamMode.STATIC, hw=hw)
        result = runner.run_forward()
        assert isinstance(result, LayerStreamResult)
        assert result.mode == StreamMode.STATIC
        assert result.total_time_ms > 0
        assert result.overlap_saved_ms == 0.0
        runner.shutdown()

    def test_stream_1buf_forward(self, small_config, hw):
        runner = LayerStreamRunner(config=small_config, mode=StreamMode.STREAM_1BUF, hw=hw)
        result = runner.run_forward()
        assert isinstance(result, LayerStreamResult)
        assert result.mode == StreamMode.STREAM_1BUF
        assert result.peak_memory_bytes == small_config.layer_size_bytes
        assert result.total_time_ms > 0
        runner.shutdown()

    def test_pingpong_forward(self, small_config, hw):
        runner = LayerStreamRunner(config=small_config, mode=StreamMode.PINGPONG_2BUF, hw=hw)
        result = runner.run_forward()
        assert isinstance(result, LayerStreamResult)
        assert result.mode == StreamMode.PINGPONG_2BUF
        assert result.peak_memory_bytes == 2 * small_config.layer_size_bytes
        assert result.overlap_saved_ms >= 0.0
        runner.shutdown()

    def test_memory_ordering_across_runners(self, small_config, hw):
        results = {}
        for mode in StreamMode:
            runner = LayerStreamRunner(config=small_config, mode=mode, hw=hw)
            results[mode] = runner.run_forward()
            runner.shutdown()

        assert results[StreamMode.STREAM_1BUF].peak_memory_bytes < results[StreamMode.PINGPONG_2BUF].peak_memory_bytes
        assert results[StreamMode.PINGPONG_2BUF].peak_memory_bytes < results[StreamMode.STATIC].peak_memory_bytes

    def test_custom_input(self, small_config, hw):
        runner = LayerStreamRunner(config=small_config, mode=StreamMode.STREAM_1BUF, hw=hw)
        dim = runner._layer_dim
        x = np.ones(dim, dtype=np.float32)
        result = runner.run_forward(x)
        assert result.total_time_ms > 0
        runner.shutdown()

    def test_all_modes_produce_similar_time(self, small_config, hw):
        times = {}
        for mode in StreamMode:
            runner = LayerStreamRunner(config=small_config, mode=mode, hw=hw)
            result = runner.run_forward()
            times[mode] = result.total_time_ms
            runner.shutdown()

        max_time = max(times.values())
        min_time = min(times.values())
        assert max_time < min_time * 3.0


# ---------------------------------------------------------------------------
# StreamMode enum
# ---------------------------------------------------------------------------

class TestStreamMode:
    def test_values(self):
        assert StreamMode.STATIC.value == "static"
        assert StreamMode.STREAM_1BUF.value == "stream_1buf"
        assert StreamMode.PINGPONG_2BUF.value == "pingpong_2buf"

    def test_iteration(self):
        modes = list(StreamMode)
        assert len(modes) == 3


# ---------------------------------------------------------------------------
# LayerStreamConfig properties
# ---------------------------------------------------------------------------

class TestLayerStreamConfig:
    def test_total_model_bytes(self, config):
        assert config.total_model_bytes == config.num_layers * config.layer_size_bytes

    def test_layer_size_mb(self, config):
        assert config.layer_size_mb == config.layer_size_bytes / (1024 * 1024)

    def test_elements_per_layer(self, config):
        assert config.elements_per_layer == config.layer_size_bytes // config.dtype_bytes

    def test_layer_dim(self, config):
        dim = config.layer_dim
        assert dim * dim <= config.elements_per_layer
