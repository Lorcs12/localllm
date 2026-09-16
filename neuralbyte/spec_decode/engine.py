"""Speculative decoding engine with SMW-adapted draft head (Adaptive-EAGLE).

Orchestrates the three-phase inference loop:

  Phase 1 — Speculative Draft: The frozen draft model extracts features,
  the Ridge head predicts K tokens autoregressively.

  Phase 2 — Frontier Verification: The target model ingests the K draft
  tokens in a single parallel batch, accepts the first M, rejects token M+1,
  and outputs its ground-truth token T.

  Phase 3 — Zero-Gradient Update: The rejected token triggers the trace-
  bounded SMW update on the covariance, then a sparse margin update on W:
  boost W[T], penalize W[P]. Resume drafting with the corrected head.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import OSDConfig
from .draft_head import SMWDraftHead
from .features import extract_draft_features, get_lm_head_weights, last_token_features
from .metrics import AcceptanceTracker


@dataclass
class VerificationResult:
    """Output of one verification round."""

    accepted_tokens: list[int]
    n_drafted: int
    n_accepted: int
    training_pairs: list[tuple[np.ndarray, int, int]]  # (features, target_id, drafted_id)
    resampled_token: int | None


@dataclass
class GenerationResult:
    """Output of a full generation run."""

    text: str
    token_ids: list[int]
    n_rounds: int
    total_drafted: int
    total_accepted: int
    acceptance_rate: float
    acceptance_curve: list[float]
    wall_time_s: float
    updates_applied: int
    tokens_per_second: float
    tracker: AcceptanceTracker


@dataclass
class SpeculativeEngine:
    """Orchestrates the draft-verify-update loop for Adaptive-EAGLE.

    Neither model is modified. The only mutable state is the SMWDraftHead,
    which lives in numpy space between the two torch models.
    """

    draft_model: Any
    target_model: Any
    draft_head: SMWDraftHead
    tokenizer: Any
    config: OSDConfig
    tracker: AcceptanceTracker = field(default_factory=AcceptanceTracker)
    _device: str = "cpu"

    @classmethod
    def from_models(
        cls,
        draft_model_name: str,
        target_model_name: str,
        config: OSDConfig | None = None,
        device: str = "auto",
    ) -> SpeculativeEngine:
        """Load both models and initialize the draft head from the draft model's LM head."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config is None:
            config = OSDConfig()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name).to(device)
        draft_model.eval()
        for p in draft_model.parameters():
            p.requires_grad_(False)

        if target_model_name == draft_model_name:
            target_model = draft_model
        else:
            target_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(device)
            target_model.eval()
            for p in target_model.parameters():
                p.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        W_lm = get_lm_head_weights(draft_model)
        draft_head = SMWDraftHead.from_lm_head(W_lm, config)

        return cls(
            draft_model=draft_model,
            target_model=target_model,
            draft_head=draft_head,
            tokenizer=tokenizer,
            config=config,
            _device=device,
        )

    def generate(self, prompt: str, max_new_tokens: int = 100) -> GenerationResult:
        """Run the full speculative decoding loop with SMW adaptation."""
        t0 = time.perf_counter()

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        generated: list[int] = []
        n_rounds = 0
        total_drafted = 0
        total_accepted = 0
        total_updates = 0
        per_round_rates: list[float] = []

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            k = min(self.config.draft_length, remaining)
            if k <= 0:
                break

            # --- Phase 1: Speculative Draft ---
            draft_tokens, draft_features, draft_probs = self._draft_tokens(
                input_ids, attention_mask, k
            )

            # --- Phase 2: Frontier Verification ---
            result = self._verify_tokens(
                input_ids, attention_mask, draft_tokens, draft_features, draft_probs
            )

            # --- Phase 3: Zero-Gradient Update ---
            n_updates = self._adapt_head(result.training_pairs)

            # Append accepted tokens + resampled token
            accepted = result.accepted_tokens
            if result.resampled_token is not None:
                accepted = accepted + [result.resampled_token]

            for tok in accepted:
                if len(generated) >= max_new_tokens:
                    break
                generated.append(tok)

            # Extend input_ids for next round
            if accepted:
                new_ids = torch.tensor([accepted], device=self._device)
                input_ids = torch.cat([input_ids, new_ids], dim=1)
                new_mask = torch.ones(1, len(accepted), device=self._device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)

            rate = result.n_accepted / result.n_drafted if result.n_drafted > 0 else 0.0
            per_round_rates.append(rate)
            total_drafted += result.n_drafted
            total_accepted += result.n_accepted
            total_updates += n_updates
            self.tracker.record_round(result.n_drafted, result.n_accepted, n_updates)
            n_rounds += 1

            # Stop on EOS
            if generated and generated[-1] == self.tokenizer.eos_token_id:
                break

        elapsed = time.perf_counter() - t0
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return GenerationResult(
            text=text,
            token_ids=generated,
            n_rounds=n_rounds,
            total_drafted=total_drafted,
            total_accepted=total_accepted,
            acceptance_rate=total_accepted / total_drafted if total_drafted > 0 else 0.0,
            acceptance_curve=per_round_rates,
            wall_time_s=elapsed,
            updates_applied=total_updates,
            tokens_per_second=len(generated) / elapsed if elapsed > 0 else 0.0,
            tracker=self.tracker,
        )

    def _draft_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        k: int,
    ) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
        """Phase 1: autoregressively generate K draft tokens using the Ridge head."""
        draft_tokens: list[int] = []
        draft_features: list[np.ndarray] = []
        draft_probs: list[np.ndarray] = []

        cur_ids = input_ids.clone()
        cur_mask = attention_mask.clone()

        for _ in range(k):
            features, _ = extract_draft_features(
                self.draft_model, cur_ids, cur_mask, self.config
            )
            feat = features[0]  # [D], single sequence
            probs = self.draft_head.predict_probs(feat)

            token_id = int(np.argmax(probs))
            draft_tokens.append(token_id)
            draft_features.append(feat)
            draft_probs.append(probs)

            next_id = torch.tensor([[token_id]], device=cur_ids.device)
            cur_ids = torch.cat([cur_ids, next_id], dim=1)
            next_mask = torch.ones(1, 1, device=cur_mask.device, dtype=cur_mask.dtype)
            cur_mask = torch.cat([cur_mask, next_mask], dim=1)

        return draft_tokens, draft_features, draft_probs

    def _verify_tokens(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        draft_tokens: list[int],
        draft_features: list[np.ndarray],
        draft_probs: list[np.ndarray],
    ) -> VerificationResult:
        """Phase 2: verify draft tokens with the target model in one batch pass."""
        k = len(draft_tokens)

        # Build input: prefix + all draft tokens
        draft_ids = torch.tensor([draft_tokens], device=prefix_ids.device)
        full_ids = torch.cat([prefix_ids, draft_ids], dim=1)
        full_mask = torch.cat([
            prefix_mask,
            torch.ones(1, k, device=prefix_mask.device, dtype=prefix_mask.dtype),
        ], dim=1)

        # Single forward pass through target model
        with torch.inference_mode():
            target_out = self.target_model(
                input_ids=full_ids,
                attention_mask=full_mask,
                output_hidden_states=True,
            )

        target_logits = target_out.logits[0]  # [seq_len, V]
        prefix_len = prefix_ids.shape[1]

        accepted: list[int] = []
        training_pairs: list[tuple[np.ndarray, int, int]] = []

        for i in range(k):
            # Target's prediction at the position just before draft token i
            target_pos = prefix_len + i - 1
            target_token_logits = target_logits[target_pos].float()
            target_token_probs = torch.softmax(target_token_logits, dim=-1).cpu().numpy()

            draft_token = draft_tokens[i]
            p_target = float(target_token_probs[draft_token])
            p_draft = float(draft_probs[i][draft_token])

            # Standard speculative decoding acceptance criterion
            if p_draft > 0:
                accept_prob = min(1.0, p_target / p_draft)
            else:
                accept_prob = 1.0 if p_target > 0 else 0.0

            if np.random.random() < accept_prob:
                accepted.append(draft_token)
            else:
                # Rejection: collect training pair and resample from residual
                target_id = int(np.argmax(target_token_probs))

                # Extract target model's features at this position for the update
                target_features = last_token_features(
                    target_out.hidden_states,
                    full_mask,
                    layer_index=self.config.layer_index,
                    use_layer_norm=self.config.use_layer_norm,
                )
                training_pairs.append((
                    draft_features[i],
                    target_id,
                    draft_token,
                ))

                # Resample from adjusted distribution (standard spec decode)
                residual = np.maximum(target_token_probs - draft_probs[i], 0.0)
                residual_sum = residual.sum()
                if residual_sum > 0:
                    residual /= residual_sum
                    resampled = int(np.random.choice(len(residual), p=residual))
                else:
                    resampled = target_id

                return VerificationResult(
                    accepted_tokens=accepted,
                    n_drafted=k,
                    n_accepted=len(accepted),
                    training_pairs=training_pairs,
                    resampled_token=resampled,
                )

        # All tokens accepted — sample one bonus token from target's last position
        last_target_logits = target_logits[prefix_len + k - 1].float()
        last_target_probs = torch.softmax(last_target_logits, dim=-1).cpu().numpy()
        bonus_token = int(np.argmax(last_target_probs))

        return VerificationResult(
            accepted_tokens=accepted,
            n_drafted=k,
            n_accepted=k,
            training_pairs=training_pairs,
            resampled_token=bonus_token,
        )

    def _adapt_head(self, training_pairs: list[tuple[np.ndarray, int, int]]) -> int:
        """Phase 3: fold rejections into the draft head via sparse margin updates."""
        for features, target_id, drafted_id in training_pairs:
            self.draft_head.update(features, target_id, drafted_token_id=drafted_id)
        return len(training_pairs)
