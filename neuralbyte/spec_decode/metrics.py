"""Acceptance rate tracking and adaptation diagnostics for SMW speculative decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AcceptanceTracker:
    """Tracks per-round acceptance rates to measure whether the SMW draft head
    is adapting to the target model's distribution during generation.

    The key signal: if adaptation is working, acceptance_rate should climb
    over time. A flat curve means the head is not learning (or draft == target).
    """

    _rounds: list[dict[str, int]] = field(default_factory=list)

    def record_round(self, n_drafted: int, n_accepted: int, n_updates: int = 0) -> None:
        self._rounds.append({
            "n_drafted": n_drafted,
            "n_accepted": n_accepted,
            "n_updates": n_updates,
        })

    @property
    def n_rounds(self) -> int:
        return len(self._rounds)

    def acceptance_rate(self) -> float:
        total_d = sum(r["n_drafted"] for r in self._rounds)
        total_a = sum(r["n_accepted"] for r in self._rounds)
        return total_a / total_d if total_d > 0 else 0.0

    def recent_acceptance_rate(self, window: int = 10) -> float:
        recent = self._rounds[-window:]
        total_d = sum(r["n_drafted"] for r in recent)
        total_a = sum(r["n_accepted"] for r in recent)
        return total_a / total_d if total_d > 0 else 0.0

    def adaptation_gain(self) -> float | None:
        """Compare acceptance rate of last 25% of rounds vs first 25%.

        Positive means the head is learning. None if fewer than 8 rounds.
        """
        if len(self._rounds) < 8:
            return None
        q = len(self._rounds) // 4
        first = self._rounds[:q]
        last = self._rounds[-q:]

        def _rate(rounds: list[dict]) -> float:
            d = sum(r["n_drafted"] for r in rounds)
            a = sum(r["n_accepted"] for r in rounds)
            return a / d if d > 0 else 0.0

        return _rate(last) - _rate(first)

    def per_round_rates(self) -> list[float]:
        return [
            r["n_accepted"] / r["n_drafted"] if r["n_drafted"] > 0 else 0.0
            for r in self._rounds
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rounds": self.n_rounds,
            "acceptance_rate": round(self.acceptance_rate(), 4),
            "recent_acceptance_rate": round(self.recent_acceptance_rate(), 4),
            "adaptation_gain": (
                round(self.adaptation_gain(), 4)
                if self.adaptation_gain() is not None
                else None
            ),
            "total_drafted": sum(r["n_drafted"] for r in self._rounds),
            "total_accepted": sum(r["n_accepted"] for r in self._rounds),
            "total_updates": sum(r["n_updates"] for r in self._rounds),
        }

    def summary(self) -> str:
        d = self.to_dict()
        gain = d["adaptation_gain"]
        gain_str = f", gain={gain:+.2%}" if gain is not None else ""
        return (
            f"accept={d['acceptance_rate']:.1%} "
            f"({d['total_accepted']}/{d['total_drafted']} over {d['n_rounds']} rounds"
            f"{gain_str})"
        )
