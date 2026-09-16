"""Tests for NeuronMap, CUP predictor, and slab fetch subsystems."""
import numpy as np
import pytest

from neuralbyte.spec_decode.neuron_map import NeuronMap, NeuronMapConfig, SlabIndex
from neuralbyte.spec_decode.predictor import CUPConfig, CUPredictor, PredictionResult, evaluate_prediction
from neuralbyte.spec_decode.slab_fetch import SlabFetcher, FetchResult, PipelineTimer


# ---------------------------------------------------------------------------
# NeuronMap
# ---------------------------------------------------------------------------

class TestNeuronMap:
    def test_hot_cold_split_ratio(self):
        config = NeuronMapConfig(n_neurons=100, hot_ratio=0.20)
        freqs = np.random.RandomState(0).rand(100)
        nmap = NeuronMap.from_activation_frequencies(freqs, config)

        assert nmap.n_hot == 20
        assert nmap.n_cold == 80
        assert nmap.n_hot + nmap.n_cold == 100

    def test_hot_neurons_have_highest_frequency(self):
        config = NeuronMapConfig(n_neurons=50, hot_ratio=0.20)
        freqs = np.arange(50, dtype=np.float64)
        nmap = NeuronMap.from_activation_frequencies(freqs, config)

        min_hot_freq = freqs[nmap.hot_indices].min()
        max_cold_freq = freqs[nmap.cold_indices].max()
        assert min_hot_freq >= max_cold_freq

    def test_masks_are_complementary(self):
        config = NeuronMapConfig(n_neurons=100, hot_ratio=0.20)
        freqs = np.random.RandomState(1).rand(100)
        nmap = NeuronMap.from_activation_frequencies(freqs, config)

        assert np.all(nmap.hot_mask == ~nmap.cold_mask)

    def test_cold_to_local_roundtrip(self):
        config = NeuronMapConfig(n_neurons=100, hot_ratio=0.20)
        freqs = np.random.RandomState(2).rand(100)
        nmap = NeuronMap.from_activation_frequencies(freqs, config)

        cold_globals = nmap.cold_indices[:5]
        local_ids = nmap.cold_to_local(cold_globals)
        roundtrip = nmap.local_to_cold(local_ids)
        np.testing.assert_array_equal(roundtrip, cold_globals)


# ---------------------------------------------------------------------------
# SlabIndex
# ---------------------------------------------------------------------------

class TestSlabIndex:
    def test_uniform_covers_all_neurons(self):
        config = NeuronMapConfig(
            slab_size_bytes=2 * 1024 * 1024,
            neuron_size_bytes=8 * 1024,
        )
        n_cold = 80
        slab = SlabIndex.uniform(n_cold, config)

        all_neurons = set()
        for neurons in slab.slab_to_neurons.values():
            all_neurons.update(neurons.tolist())
        assert len(all_neurons) == n_cold

    def test_uniform_slab_assignment(self):
        config = NeuronMapConfig(
            slab_size_bytes=2 * 1024 * 1024,
            neuron_size_bytes=8 * 1024,
        )
        n_cold = 80
        slab = SlabIndex.uniform(n_cold, config)
        neurons_per_slab = config.slab_size_bytes // config.neuron_size_bytes

        assert slab.neurons_per_slab == neurons_per_slab
        assert all(slab.neuron_to_slab[i] >= 0 for i in range(n_cold))

    def test_coactivation_clusters(self):
        config = NeuronMapConfig(
            slab_size_bytes=16 * 1024,
            neuron_size_bytes=8 * 1024,
        )
        n_cold = 10
        coact = np.zeros((n_cold, n_cold), dtype=np.float64)
        coact[0, 1] = coact[1, 0] = 100
        coact[2, 3] = coact[3, 2] = 100
        coact[4, 5] = coact[5, 4] = 100

        slab = SlabIndex.from_coactivation(coact, config)

        assert slab.neuron_to_slab[0] == slab.neuron_to_slab[1]
        assert slab.neuron_to_slab[2] == slab.neuron_to_slab[3]

    def test_neurons_to_slabs(self):
        config = NeuronMapConfig(
            slab_size_bytes=2 * 1024 * 1024,
            neuron_size_bytes=8 * 1024,
        )
        slab = SlabIndex.uniform(80, config)
        slab_ids = slab.neurons_to_slabs(np.array([0, 1, 2]))
        assert len(slab_ids) >= 1

    def test_expand_slabs(self):
        config = NeuronMapConfig(
            slab_size_bytes=2 * 1024 * 1024,
            neuron_size_bytes=8 * 1024,
        )
        slab = SlabIndex.uniform(80, config)
        slab_ids = slab.neurons_to_slabs(np.array([0]))
        expanded = slab.expand_slabs(slab_ids)
        assert 0 in expanded

    def test_fetch_stats(self):
        config = NeuronMapConfig(
            slab_size_bytes=2 * 1024 * 1024,
            neuron_size_bytes=8 * 1024,
        )
        slab = SlabIndex.uniform(800, config)
        requested = np.arange(200, dtype=np.int32)
        stats = slab.fetch_stats(requested)
        assert "n_neurons_requested" in stats
        assert stats["n_neurons_requested"] == 200
        assert stats["speedup"] > 1.0


# ---------------------------------------------------------------------------
# CUPredictor
# ---------------------------------------------------------------------------

class TestCUPredictor:
    def test_synthetic_predict_shape(self):
        cup = CUPredictor.synthetic(n_cold=80, context_dim=32)
        x = np.random.RandomState(0).randn(32).astype(np.float32)
        result = cup.predict(x)

        assert result.cold_probabilities.shape == (80,)
        assert result.predicted_cold_mask.shape == (80,)
        assert np.all((result.cold_probabilities >= 0) & (result.cold_probabilities <= 1))

    def test_predict_from_ground_truth_high_recall(self):
        n_cold = 200
        cup = CUPredictor.synthetic(n_cold=n_cold, context_dim=32)

        ground_truth = np.zeros(n_cold, dtype=np.float64)
        ground_truth[:30] = 1.0

        result = cup.predict_from_ground_truth(ground_truth, noise_std=0.05)
        metrics = evaluate_prediction(result, ground_truth)

        assert metrics["recall"] >= 0.90

    def test_evaluate_prediction_perfect(self):
        n_cold = 50
        probs = np.zeros(n_cold, dtype=np.float64)
        probs[:10] = 0.9
        mask = probs >= 0.35
        ids = np.where(mask)[0].astype(np.int32)

        result = PredictionResult(
            cold_probabilities=probs,
            predicted_cold_mask=mask,
            predicted_cold_ids=ids,
            n_predicted=len(ids),
            threshold=0.35,
        )

        truth = np.zeros(n_cold, dtype=np.float64)
        truth[:10] = 1.0
        metrics = evaluate_prediction(result, truth)

        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["pipeline_ok"] is True

    def test_evaluate_prediction_with_misses(self):
        n_cold = 50
        probs = np.zeros(n_cold, dtype=np.float64)
        probs[:5] = 0.9
        mask = probs >= 0.35
        ids = np.where(mask)[0].astype(np.int32)

        result = PredictionResult(
            cold_probabilities=probs,
            predicted_cold_mask=mask,
            predicted_cold_ids=ids,
            n_predicted=len(ids),
            threshold=0.35,
        )

        truth = np.zeros(n_cold, dtype=np.float64)
        truth[:10] = 1.0
        metrics = evaluate_prediction(result, truth)

        assert metrics["recall"] == 0.5
        assert metrics["false_negatives_cache_misses"] == 5
        assert metrics["pipeline_ok"] is False


# ---------------------------------------------------------------------------
# SlabFetcher
# ---------------------------------------------------------------------------

class TestSlabFetcher:
    def _make_neuron_map(self):
        config = NeuronMapConfig(
            n_neurons=100,
            hot_ratio=0.20,
            slab_size_bytes=16 * 1024,
            neuron_size_bytes=8 * 1024,
        )
        freqs = np.random.RandomState(0).rand(100)
        return NeuronMap.from_activation_frequencies(freqs, config)

    def test_simulated_sync_fetch(self):
        nmap = self._make_neuron_map()
        fetcher = SlabFetcher.simulated(nmap, neuron_dim=64)

        cup = CUPredictor.synthetic(n_cold=nmap.n_cold, context_dim=32)
        x = np.random.RandomState(0).randn(32).astype(np.float32)
        prediction = cup.predict(x)

        result = fetcher.fetch_slabs_sync(prediction)
        assert isinstance(result, FetchResult)
        assert result.sequential is True
        assert result.n_slabs >= 1
        assert result.fetch_time_ms >= 0
        fetcher.shutdown()

    def test_simulated_async_fetch(self):
        nmap = self._make_neuron_map()
        fetcher = SlabFetcher.simulated(nmap, neuron_dim=64)

        cup = CUPredictor.synthetic(n_cold=nmap.n_cold, context_dim=32)
        x = np.random.RandomState(0).randn(32).astype(np.float32)
        prediction = cup.predict(x)

        future = fetcher.fetch_slabs_async(prediction)
        result = future.result(timeout=10)
        assert isinstance(result, FetchResult)
        assert result.sequential is True
        fetcher.shutdown()

    def test_random_fetch_baseline(self):
        nmap = self._make_neuron_map()
        fetcher = SlabFetcher.simulated(nmap, neuron_dim=64)

        cold_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        future = fetcher.fetch_random_async(cold_ids)
        result = future.result(timeout=10)
        assert result.sequential is False
        assert result.n_neurons_loaded == 5
        fetcher.shutdown()


# ---------------------------------------------------------------------------
# PipelineTimer
# ---------------------------------------------------------------------------

class TestPipelineTimer:
    def test_record_and_total(self):
        timer = PipelineTimer()
        timer.record("predict", 2.0)
        timer.record("draft", 80.0)
        timer.record("verify", 20.0)
        assert timer.total_ms() == 102.0

    def test_to_dict(self):
        timer = PipelineTimer()
        timer.record("predict", 2.0)
        d = timer.to_dict()
        assert d["predict"] == 2.0
        assert d["total"] == 2.0

    def test_summary(self):
        timer = PipelineTimer()
        timer.record("predict", 2.0)
        s = timer.summary()
        assert "predict" in s
