from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def van_der_corput(n: int, base: int = 2) -> NDArray[np.float64]:
    """
    Generate n elements of a van der Corput sequence in base 'base'.
    """
    # Indices
    k = np.arange(n)
    vdc = np.zeros(n, dtype=float)
    denom = 1.0

    while np.any(k > 0):
        k, remainder = divmod(k, base)
        denom *= base
        vdc += remainder / denom

    return vdc


def vdc_permutation(n: int, base: int = 2) -> NDArray[np.int64]:
    """
    Generate a permutation of n elements based on the van der Corput sequence.
    """
    vdc_sequence = van_der_corput(n, base)
    return np.argsort(vdc_sequence)


class IDWInterpolationModel:
    """
    Predict values between known points using inverse-distance weighted (IDW) interpolation.
    """

    def __init__(self, k: int = 4, epsilon: float = 1e-6) -> None:
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.k = k
        self.epsilon = epsilon

    def update(self, x: float, y: float) -> None:
        """Record an observed (x, y) pair from a successful measurement."""
        self.xs.append(x)
        self.ys.append(y)

    def predict(self, x: float) -> float:
        """Predict the value at a given x via IDW interpolation."""
        n = len(self.xs)
        if n == 0:
            return 0.0
        if n == 1:
            return self.ys[0]

        xs = np.array(self.xs)
        es = np.array(self.ys)

        if n == 2:
            x0, x1 = xs
            e0, e1 = es
            dx = x1 - x0
            if abs(dx) < self.epsilon:
                return float(0.5 * (e0 + e1))
            t = (x - x0) / dx
            return float(e0 + t * (e1 - e0))

        distances = np.abs(xs - x)
        nearest_idx = np.argsort(distances)[: self.k]
        d = distances[nearest_idx]
        e = es[nearest_idx]

        weights = 1.0 / (d + self.epsilon)
        return float(np.sum(weights * e) / np.sum(weights))
