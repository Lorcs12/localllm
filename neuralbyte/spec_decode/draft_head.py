"""Self-adapting Ridge draft head for speculative decoding (Adaptive-EAGLE).

The mathematical core: a next-token prediction head that absorbs target-model
corrections via trace-bounded Sherman-Morrison-Woodbury updates, without
gradients or optimizer state.

Two key innovations over naive SMW:

1. Trace-bounded covariance: unchecked rank-1 updates cause the inverse
   covariance trace to grow without bound (3.4B+ at production dimensionality).
   A forgetting factor (lambda_forget) decays old curvature each step, and a
   hard trace cap rescales when the accumulated trace exceeds max_trace.
   This guarantees the head can run indefinitely without numerical blowup.

2. Sparse margin update: instead of updating the full W [D, V] via a one-hot
   outer product, each rejection updates exactly two rows of W:
     W[target]  += x_adj   (boost the correct answer)
     W[drafted] -= x_adj   (penalize the mistake)
   This is O(D) per rejection regardless of vocabulary size.

The SMW formula for Ainv is inlined from ridge.rank_one_update (ridge.py) with
the trace bound added. The downdate for sliding-window eviction uses the same
formula with negated weight (ridge.rank_one_downdate).
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..ridge import invert as ridge_invert
from .config import OSDConfig


@dataclass
class SMWDraftHead:
    """Ridge regression head for next-token prediction, adapted online via
    trace-bounded Sherman-Morrison rank-1 updates.

    Initialized from a draft model's LM head weights so it produces identical
    predictions at token 0. As target-model rejections are folded in via
    sparse margin updates, the head migrates toward the target's distribution.

    Keeps W [D, V] explicitly and updates only 2 rows per rejection.
    Keeps (Ainv [D, D], XtWX [D, D]) for the trace-bounded covariance loop.
    """

    W: np.ndarray
    Ainv: np.ndarray
    XtWX: np.ndarray
    config: OSDConfig
    _window: deque = field(default_factory=deque)
    _update_count: int = 0
    _vocab_size: int = 0
    _feature_dim: int = 0

    @classmethod
    def from_lm_head(cls, W_lm_head: np.ndarray, config: OSDConfig | None = None) -> SMWDraftHead:
        """Initialize from the draft model's LM head [D, V].

        The ridge head IS the original LM head at token 0. XtWX starts as
        lam*I (no data, only the regularizer). Ainv = (1/lam)*I.
        """
        if config is None:
            config = OSDConfig()
        W = np.asarray(W_lm_head, dtype=np.float64).copy()
        D, V = W.shape
        lam = config.lam

        XtWX = lam * np.eye(D, dtype=np.float64)
        Ainv = (1.0 / lam) * np.eye(D, dtype=np.float64)

        return cls(
            W=W,
            Ainv=Ainv,
            XtWX=XtWX,
            config=config,
            _window=deque(maxlen=config.window_size),
            _update_count=0,
            _vocab_size=V,
            _feature_dim=D,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Compute logits: features @ W.T / temperature.

        W is [D, V] so W.T is [V, D]; features [batch, D] @ W gives [batch, V].
        Returns raw logits, NOT softmaxed.
        """
        x = np.asarray(features, dtype=np.float64)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        logits = x @ self.W / self.config.temperature  # [batch, V]
        if squeeze:
            logits = logits[0]
        return logits

    def predict_probs(self, features: np.ndarray) -> np.ndarray:
        """Logits -> softmax probabilities."""
        logits = self.predict(features)
        squeeze = logits.ndim == 1
        if squeeze:
            logits = logits[None, :]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        if squeeze:
            probs = probs[0]
        return probs

    def update(
        self,
        features: np.ndarray,
        target_token_id: int,
        drafted_token_id: int | None = None,
    ) -> None:
        """Absorb one target-model correction via trace-bounded SMW + sparse margin.

        Phase 1 — Trace-bounded covariance update (O(D^2)):
          Apply forgetting factor to decay old curvature, then rank-1 SMW
          update on Ainv. If trace(Ainv) exceeds max_trace, rescale to cap it.
          The adjusted feature vector x_adj absorbs the rescaling.

        Phase 2 — Sparse margin update on W (O(D)):
          W[target]  += x_adj   (boost the correct answer)
          W[drafted] -= x_adj   (penalize the mistake)
          Only 2 rows of W change, regardless of vocabulary size.
        """
        x = np.asarray(features, dtype=np.float64).ravel()
        w = self.config.update_weight

        if len(self._window) >= self.config.window_size:
            self._evict_oldest()

        # --- Phase 1: Trace-bounded covariance update ---

        # Forgetting factor: decay accumulated curvature so old observations
        # lose influence geometrically. Without this, trace grows linearly
        # with the number of updates.
        lf = self.config.lambda_forget
        self.Ainv *= (1.0 / lf)
        self.XtWX *= lf

        # SMW rank-1 update (from ridge.rank_one_update lines 179-190)
        Ainv_x = self.Ainv @ x
        denom = 1.0 + w * float(x @ Ainv_x)
        if abs(denom) < self.config.min_denom:
            warnings.warn(f"Skipping update: denom={denom:.3e} near singular")
            return
        self.Ainv -= (w / denom) * np.outer(Ainv_x, Ainv_x)
        self.XtWX += w * np.outer(x, x)

        # Trace cap: if trace(Ainv) exceeds max_trace, rescale so the head
        # can run indefinitely without the inverse blowing up.
        trace = float(np.trace(self.Ainv))
        x_adj = x.copy()
        if trace > self.config.max_trace:
            scale = self.config.max_trace / trace
            self.Ainv *= scale
            x_adj *= np.sqrt(scale)

        # --- Phase 2: Sparse margin update on W ---

        # Boost the correct token's row
        self.W[:, target_token_id] += x_adj

        # Penalize the incorrectly drafted token's row (if provided)
        if drafted_token_id is not None and drafted_token_id != target_token_id:
            self.W[:, drafted_token_id] -= x_adj

        self._window.append((x.copy(), target_token_id, drafted_token_id, w))
        self._update_count += 1
        self._maybe_reinvert()

    def _evict_oldest(self) -> None:
        """Remove the oldest correction from the sliding window.

        Downdates the covariance (rank-1 removal) and reverses the margin
        update on W.
        """
        x_old, target_old, drafted_old, w_old = self._window.popleft()

        # Covariance downdate: same SMW formula with negated weight
        Ainv_x = self.Ainv @ x_old
        denom = 1.0 + (-w_old) * float(x_old @ Ainv_x)
        if abs(denom) < self.config.min_denom:
            warnings.warn(f"Skipping eviction: denom={denom:.3e} near singular")
            return
        self.Ainv -= (-w_old / denom) * np.outer(Ainv_x, Ainv_x)
        self.XtWX -= w_old * np.outer(x_old, x_old)

        # Reverse the margin update on W
        self.W[:, target_old] -= x_old
        if drafted_old is not None and drafted_old != target_old:
            self.W[:, drafted_old] += x_old

    def _maybe_reinvert(self) -> None:
        """Periodic exact reinversion to bound floating-point drift."""
        if self._update_count > 0 and self._update_count % self.config.reinvert_every == 0:
            self.Ainv = ridge_invert(self.XtWX)

    def leverage(self, features: np.ndarray) -> float:
        """OOD confidence signal: h(x) = x^T Ainv x.

        Same quantity drift.py::OODGate uses. High leverage means the input
        is far from the correction distribution the head has seen.
        """
        x = np.asarray(features, dtype=np.float64).ravel()
        return float(x @ (self.Ainv @ x))

    def trace(self) -> float:
        """Current trace of Ainv — the stability metric."""
        return float(np.trace(self.Ainv))

    def stats(self) -> dict:
        return {
            "feature_dim": self._feature_dim,
            "vocab_size": self._vocab_size,
            "update_count": self._update_count,
            "window_fill": len(self._window),
            "window_capacity": self.config.window_size,
            "trace": self.trace(),
        }
