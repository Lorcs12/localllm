"""Ridge regression utilities — matrix inversion for periodic reinversion."""
from __future__ import annotations

import numpy as np


def invert(XtWX: np.ndarray) -> np.ndarray:
    """Compute the exact inverse of the regularized covariance matrix.

    Used by SMWDraftHead._maybe_reinvert to bound floating-point drift
    that accumulates from repeated rank-1 SMW updates.
    """
    return np.linalg.inv(XtWX)
