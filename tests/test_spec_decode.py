"""Pure math tests for SMWDraftHead and AcceptanceTracker — no models needed.

Also includes a lightweight integration test for the verify→adapt wiring
that uses a mock target model (no real weights downloaded).
"""
import numpy as np
import pytest
import torch

from neuralbyte.spec_decode.config import OSDConfig
from neuralbyte.spec_decode.draft_head import SMWDraftHead
from neuralbyte.spec_decode.engine import SpeculativeEngine
from neuralbyte.spec_decode.metrics import AcceptanceTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    return OSDConfig(
        lam=1.0,
        temperature=1.0,
        window_size=16,
        update_weight=1.0,
        lambda_forget=0.995,
        max_trace=500.0,
        draft_length=5,
        reinvert_every=20,
    )


@pytest.fixture
def head(small_config):
    rng = np.random.RandomState(0)
    W_lm = rng.randn(32, 64).astype(np.float64)
    return SMWDraftHead.from_lm_head(W_lm, small_config)


# ---------------------------------------------------------------------------
# SMWDraftHead — initialization
# ---------------------------------------------------------------------------

class TestSMWDraftHeadInit:
    def test_shapes(self, head):
        assert head.W.shape == (32, 64)
        assert head.Ainv.shape == (32, 32)
        assert head.XtWX.shape == (32, 32)

    def test_ainv_is_identity_scaled(self, head):
        expected = np.eye(32) / head.config.lam
        np.testing.assert_allclose(head.Ainv, expected)

    def test_xtwx_is_identity_scaled(self, head):
        expected = head.config.lam * np.eye(32)
        np.testing.assert_allclose(head.XtWX, expected)

    def test_update_count_zero(self, head):
        assert head._update_count == 0
        assert len(head._window) == 0


# ---------------------------------------------------------------------------
# SMWDraftHead — predict
# ---------------------------------------------------------------------------

class TestSMWDraftHeadPredict:
    def test_logits_shape_1d(self, head):
        x = np.random.randn(32)
        logits = head.predict(x)
        assert logits.shape == (64,)

    def test_logits_shape_batch(self, head):
        x = np.random.randn(3, 32)
        logits = head.predict(x)
        assert logits.shape == (3, 64)

    def test_logits_match_manual(self, head):
        x = np.random.RandomState(1).randn(32)
        logits = head.predict(x)
        expected = x @ head.W / head.config.temperature
        np.testing.assert_allclose(logits, expected)

    def test_probs_sum_to_one(self, head):
        x = np.random.randn(32)
        probs = head.predict_probs(x)
        assert probs.shape == (64,)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)

    def test_probs_non_negative(self, head):
        x = np.random.randn(32)
        probs = head.predict_probs(x)
        assert np.all(probs >= 0)


# ---------------------------------------------------------------------------
# SMWDraftHead — update
# ---------------------------------------------------------------------------

class TestSMWDraftHeadUpdate:
    def test_sparse_margin_boosts_target(self, head):
        x = np.random.RandomState(2).randn(32)
        target_col_before = head.W[:, 10].copy()
        head.update(x, target_token_id=10, drafted_token_id=20)
        assert np.any(head.W[:, 10] != target_col_before)

    def test_sparse_margin_penalizes_drafted(self, head):
        x = np.random.RandomState(2).randn(32)
        drafted_col_before = head.W[:, 20].copy()
        head.update(x, target_token_id=10, drafted_token_id=20)
        assert np.any(head.W[:, 20] != drafted_col_before)

    def test_other_cols_unchanged(self, head):
        x = np.random.RandomState(2).randn(32)
        col5_before = head.W[:, 5].copy()
        head.update(x, target_token_id=10, drafted_token_id=20)
        np.testing.assert_array_equal(head.W[:, 5], col5_before)

    def test_same_target_drafted_no_penalty(self, head):
        x = np.random.RandomState(3).randn(32)
        W_before = head.W.copy()
        head.update(x, target_token_id=10, drafted_token_id=10)
        # Only target column should change (boost), no penalty
        changed_cols = np.where(np.any(head.W != W_before, axis=0))[0]
        assert list(changed_cols) == [10]

    def test_update_count_increments(self, head):
        x = np.random.randn(32)
        head.update(x, 0, 1)
        head.update(x, 0, 1)
        assert head._update_count == 2

    def test_window_fills(self, head):
        x = np.random.randn(32)
        for _ in range(5):
            head.update(x, 0, 1)
        assert len(head._window) == 5


# ---------------------------------------------------------------------------
# SMWDraftHead — trace bounding
# ---------------------------------------------------------------------------

class TestTraceBounding:
    def test_trace_stays_bounded(self):
        config = OSDConfig(
            lam=1.0,
            max_trace=100.0,
            window_size=256,
            reinvert_every=1000,
        )
        rng = np.random.RandomState(42)
        W_lm = rng.randn(16, 32).astype(np.float64)
        head = SMWDraftHead.from_lm_head(W_lm, config)

        for i in range(200):
            x = rng.randn(16) * 3.0
            head.update(x, target_token_id=i % 32, drafted_token_id=(i + 1) % 32)

        assert head.trace() <= config.max_trace * 1.1


# ---------------------------------------------------------------------------
# SMWDraftHead — sliding window eviction
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_window_evicts_oldest(self):
        config = OSDConfig(window_size=4, reinvert_every=1000)
        rng = np.random.RandomState(7)
        W_lm = rng.randn(8, 16).astype(np.float64)
        head = SMWDraftHead.from_lm_head(W_lm, config)

        for i in range(6):
            x = rng.randn(8)
            head.update(x, target_token_id=i % 16, drafted_token_id=(i + 1) % 16)

        assert len(head._window) == 4
        assert head._update_count == 6


# ---------------------------------------------------------------------------
# SMWDraftHead — periodic reinversion
# ---------------------------------------------------------------------------

class TestReinversion:
    def test_reinversion_resets_ainv(self):
        config = OSDConfig(reinvert_every=5, window_size=256)
        rng = np.random.RandomState(99)
        W_lm = rng.randn(8, 16).astype(np.float64)
        head = SMWDraftHead.from_lm_head(W_lm, config)

        for i in range(5):
            x = rng.randn(8)
            head.update(x, target_token_id=0, drafted_token_id=1)

        # After reinversion, Ainv should be close to inv(XtWX)
        expected_ainv = np.linalg.inv(head.XtWX)
        np.testing.assert_allclose(head.Ainv, expected_ainv, atol=1e-8)


# ---------------------------------------------------------------------------
# SMWDraftHead — leverage / stats
# ---------------------------------------------------------------------------

class TestSMWUpdateMutatesState:
    """Tripwire: verify that a rejection update actually changes W and Ainv."""

    def test_update_changes_W_and_Ainv(self):
        config = OSDConfig(
            lam=1.0, max_trace=500.0, window_size=16, reinvert_every=1000,
        )
        rng = np.random.RandomState(77)
        W_lm = rng.randn(8, 20).astype(np.float64)
        head = SMWDraftHead.from_lm_head(W_lm, config)

        x = rng.randn(8)
        W_before = head.W.copy()
        Ainv_before = head.Ainv.copy()

        head.update(x, target_token_id=3, drafted_token_id=7)

        assert not np.allclose(head.W, W_before), "W should change after update"
        assert not np.allclose(head.Ainv, Ainv_before), "Ainv should change after update"
        assert head.trace() <= config.max_trace + 1e-9, "trace must stay bounded"


class TestLeverageAndStats:
    def test_leverage_positive(self, head):
        x = np.random.randn(32)
        h = head.leverage(x)
        assert h > 0

    def test_stats_keys(self, head):
        s = head.stats()
        assert "feature_dim" in s
        assert "vocab_size" in s
        assert "update_count" in s
        assert "trace" in s


# ---------------------------------------------------------------------------
# AcceptanceTracker
# ---------------------------------------------------------------------------

class TestAcceptanceTracker:
    def test_empty_tracker(self):
        t = AcceptanceTracker()
        assert t.n_rounds == 0
        assert t.acceptance_rate() == 0.0

    def test_record_round(self):
        t = AcceptanceTracker()
        t.record_round(10, 7, 3)
        assert t.n_rounds == 1
        assert t.acceptance_rate() == 0.7

    def test_recent_acceptance_rate(self):
        t = AcceptanceTracker()
        for _ in range(5):
            t.record_round(10, 3)  # 30%
        for _ in range(5):
            t.record_round(10, 8)  # 80%
        assert t.recent_acceptance_rate(window=5) == 0.8

    def test_adaptation_gain_positive(self):
        t = AcceptanceTracker()
        for _ in range(4):
            t.record_round(10, 3)
        for _ in range(4):
            t.record_round(10, 9)
        gain = t.adaptation_gain()
        assert gain is not None
        assert gain > 0

    def test_adaptation_gain_none_when_few_rounds(self):
        t = AcceptanceTracker()
        for _ in range(3):
            t.record_round(10, 5)
        assert t.adaptation_gain() is None

    def test_per_round_rates(self):
        t = AcceptanceTracker()
        t.record_round(10, 5)
        t.record_round(10, 10)
        assert t.per_round_rates() == [0.5, 1.0]

    def test_to_dict_keys(self):
        t = AcceptanceTracker()
        t.record_round(10, 5)
        d = t.to_dict()
        assert "n_rounds" in d
        assert "acceptance_rate" in d
        assert "total_drafted" in d

    def test_summary_string(self):
        t = AcceptanceTracker()
        t.record_round(10, 7)
        s = t.summary()
        assert "accept=" in s
        assert "7/10" in s


# ---------------------------------------------------------------------------
# Integration: verify→adapt wiring (mock target model, no real weights)
# ---------------------------------------------------------------------------

class _MockOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class _MockTargetModel:
    """Target model that always predicts `preferred_id` with near-certainty."""

    def __init__(self, preferred_id: int, vocab_size: int, hidden_dim: int):
        self.preferred_id = preferred_id
        self.V = vocab_size
        self.D = hidden_dim

    def __call__(self, input_ids, attention_mask, output_hidden_states=False, **kw):
        seq_len = input_ids.shape[1]
        logits = torch.zeros(1, seq_len, self.V)
        logits[:, :, self.preferred_id] = 50.0
        hidden_states = None
        if output_hidden_states:
            hidden_states = (torch.randn(1, seq_len, self.D),)
        return _MockOutput(logits=logits, hidden_states=hidden_states)


class TestDeliberateWrongTokenRejection:
    """Feed a deliberately wrong draft token into _verify_tokens and confirm
    the full rejection→training-pair→SMW-update pipeline fires."""

    def test_wrong_draft_triggers_rejection_and_update(self):
        D, V = 16, 32
        target_id = 5
        wrong_draft_id = 10
        config = OSDConfig(
            lam=1.0, max_trace=500.0, window_size=16,
            draft_length=1, reinvert_every=1000,
            use_layer_norm=False,
        )
        rng = np.random.RandomState(42)
        head = SMWDraftHead.from_lm_head(rng.randn(D, V).astype(np.float64), config)

        engine = SpeculativeEngine(
            draft_model=None,
            target_model=_MockTargetModel(target_id, V, D),
            draft_head=head,
            tokenizer=None,
            config=config,
        )

        draft_probs = np.zeros(V)
        draft_probs[wrong_draft_id] = 1.0

        W_before = head.W.copy()
        Ainv_before = head.Ainv.copy()

        result = engine._verify_tokens(
            prefix_ids=torch.tensor([[0]]),
            prefix_mask=torch.ones(1, 1, dtype=torch.long),
            draft_tokens=[wrong_draft_id],
            draft_features=[rng.randn(D).astype(np.float64)],
            draft_probs=[draft_probs],
        )

        assert result.n_accepted == 0, "deliberately wrong token must be rejected"
        assert len(result.training_pairs) == 1, "rejection should produce a training pair"
        assert result.resampled_token is not None, "rejection should resample"

        engine._adapt_head(result.training_pairs)

        assert not np.allclose(head.W, W_before), "W should change after adaptation"
        assert not np.allclose(head.Ainv, Ainv_before), "Ainv should change after adaptation"
        assert head.trace() <= config.max_trace + 1e-9
