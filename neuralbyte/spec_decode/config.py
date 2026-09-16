"""Configuration for SMW-based Online Speculative Decoding."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OSDConfig:
    """Hyperparameters for the self-adapting SMW draft head.

    Defaults are conservative: validated at D=4096 by nb_smw_drift_test.py
    (1000 add/delete cycles, drift < 1e-8 with reinvert_every=500).
    """

    # --- Ridge head ---
    lam: float = 1.0
    temperature: float = 1.0

    # --- Adaptation ---
    window_size: int = 128
    update_weight: float = 1.0
    lambda_forget: float = 0.995
    max_trace: float = 5000.0

    # --- Speculative decoding ---
    draft_length: int = 5

    # --- Numerical safety ---
    reinvert_every: int = 200
    min_denom: float = 1e-8

    # --- Feature extraction ---
    layer_index: int = -1
    use_layer_norm: bool = True
