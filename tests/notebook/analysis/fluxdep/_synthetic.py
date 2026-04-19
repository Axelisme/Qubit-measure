"""Shared synthetic data helpers for fluxdep tests and benchmarks."""

from __future__ import annotations

import numpy as np


def synth_ABC(N: int = 10, K: int = 4, a_true: float = 1.5, seed: int = 0):
    """Generate (A, B, C) such that A[i] = |a_true * B[i,0] + C[i,0]|."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((N, K))
    C = rng.standard_normal((N, K))
    A = np.abs(a_true * B[:, 0] + C[:, 0])
    return A, B, C
