"""CUP: Co-activation Unit Predictor.

A lightweight MLP that maps the current token context to activation
probabilities for each cold neuron across the next K tokens. Runs entirely
in RAM (~2ms) as the look-ahead step before concurrent drafting + fetching.

The predictor only targets cold neurons. Hot neurons are always in RAM and
don't need prediction — they fire chaotically across all contexts, so
predicting them is both unnecessary and unreliable.

Cold neurons fire topically with high temporal locality: if a Python neuron
is active at token t, it's ~90% likely to be active at token t+1. This
makes them highly predictable, which is why the CUP achieves near-perfect
recall even with a low over-provisioning threshold (0.35).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .neuron_map import NeuronMap, NeuronMapConfig


@dataclass
class CUPConfig:
    """Configuration for the Co-activation Unit Predictor."""

    threshold: float = 0.35
    lookahead_tokens: int = 10
    hidden_dim: int = 256
    seed: int = 42


@dataclass
class PredictionResult:
    """Output of the CUP predictor for one look-ahead window."""

    cold_probabilities: np.ndarray   # [N_COLD] activation probabilities
    predicted_cold_mask: np.ndarray  # [N_COLD] bool — above threshold
    predicted_cold_ids: np.ndarray   # cold-local indices of predicted neurons
    n_predicted: int
    threshold: float

    def global_neuron_ids(self, neuron_map: NeuronMap) -> np.ndarray:
        """Convert predicted cold-local IDs to global neuron IDs."""
        return neuron_map.local_to_cold(self.predicted_cold_ids)


@dataclass
class CUPredictor:
    """Predicts which cold neurons will fire in the next K tokens.

    In production, this is a small MLP trained on the co-activation profiling
    corpus. For simulation/prototyping, a synthetic predictor with configurable
    noise is provided via `CUPredictor.synthetic()`.

    The predictor's job is to achieve >99.5% recall at the over-provisioning
    threshold — every neuron the frontier model actually needs must be fetched
    in advance. False positives (extra neurons) are cheap (just wasted SSD
    bandwidth); false negatives (missed neurons) cause synchronous stalls
    that break the pipeline.
    """

    W1: np.ndarray         # [context_dim, hidden_dim]
    b1: np.ndarray         # [hidden_dim]
    W2: np.ndarray         # [hidden_dim, n_cold]
    b2: np.ndarray         # [n_cold]
    config: CUPConfig

    @classmethod
    def synthetic(
        cls,
        n_cold: int,
        context_dim: int = 512,
        config: CUPConfig | None = None,
    ) -> CUPredictor:
        """Create a random-initialized predictor for simulation/testing."""
        if config is None:
            config = CUPConfig()
        rng = np.random.RandomState(config.seed)
        h = config.hidden_dim

        # Xavier initialization
        W1 = rng.randn(context_dim, h).astype(np.float32) * np.sqrt(2.0 / (context_dim + h))
        b1 = np.zeros(h, dtype=np.float32)
        W2 = rng.randn(h, n_cold).astype(np.float32) * np.sqrt(2.0 / (h + n_cold))
        b2 = np.zeros(n_cold, dtype=np.float32)

        return cls(W1=W1, b1=b1, W2=W2, b2=b2, config=config)

    def predict(self, context_features: np.ndarray) -> PredictionResult:
        """Predict cold neuron activation probabilities from context features.

        context_features: [context_dim] — the current token context embedding.
        Returns activation probabilities for each cold neuron.
        """
        x = np.asarray(context_features, dtype=np.float32).ravel()

        # 2-layer MLP with ReLU + sigmoid
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))  # sigmoid

        mask = probs >= self.config.threshold
        predicted_ids = np.where(mask)[0].astype(np.int32)

        return PredictionResult(
            cold_probabilities=probs,
            predicted_cold_mask=mask,
            predicted_cold_ids=predicted_ids,
            n_predicted=len(predicted_ids),
            threshold=self.config.threshold,
        )

    def predict_from_ground_truth(
        self,
        ground_truth_cold: np.ndarray,
        noise_std: float = 0.08,
        rng: np.random.RandomState | None = None,
    ) -> PredictionResult:
        """Simulate prediction from known ground truth with configurable noise.

        This is the simulation path from nb_cup_predictor_sim.py: ground-truth
        active neurons get high probabilities (0.7-1.0), inactive get low
        (0.0-0.2), plus Gaussian noise. Used for pipeline validation without
        training a real predictor.
        """
        if rng is None:
            rng = np.random.RandomState(self.config.seed)

        n_cold = len(ground_truth_cold)
        probs = np.zeros(n_cold, dtype=np.float64)

        active = ground_truth_cold.astype(bool)
        n_active = int(active.sum())
        n_inactive = n_cold - n_active

        probs[active] = rng.uniform(0.7, 1.0, size=n_active)
        probs[~active] = rng.uniform(0.0, 0.2, size=n_inactive)
        probs += rng.normal(0, noise_std, n_cold)
        probs = np.clip(probs, 0, 1)

        mask = probs >= self.config.threshold
        predicted_ids = np.where(mask)[0].astype(np.int32)

        return PredictionResult(
            cold_probabilities=probs,
            predicted_cold_mask=mask,
            predicted_cold_ids=predicted_ids,
            n_predicted=len(predicted_ids),
            threshold=self.config.threshold,
        )


def evaluate_prediction(
    prediction: PredictionResult,
    ground_truth_cold: np.ndarray,
) -> dict:
    """Evaluate predictor quality: recall, precision, wasted neurons, cache misses."""
    truth = ground_truth_cold.astype(bool)
    predicted = prediction.predicted_cold_mask

    true_positives = int(np.sum(predicted & truth))
    false_negatives = int(np.sum(~predicted & truth))
    false_positives = int(np.sum(predicted & ~truth))
    total_needed = int(np.sum(truth))

    recall = true_positives / total_needed if total_needed > 0 else 1.0
    precision = true_positives / prediction.n_predicted if prediction.n_predicted > 0 else 1.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "n_predicted": prediction.n_predicted,
        "n_needed": total_needed,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives_cache_misses": false_negatives,
        "pipeline_ok": false_negatives == 0,
    }
