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


class IDWInterpolation:
    """
    Predict values using local weighted linear regression on the nearest k points.

    For n < 3, uses constant or linear interpolation/extrapolation.
    For n >= 3, fits a weighted linear regression on the nearest k points
    (weighted by inverse distance), which naturally extrapolates the local
    slope when the query point is beyond observed data.
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

    def move(self, dy: float) -> None:
        """Shift all observed y values by a constant amount (e.g. after a frequency update)."""
        self.ys = [y + dy for y in self.ys]

    def predict(self, x: float) -> float:
        """Predict the value at a given x via local weighted linear regression."""
        n = len(self.xs)
        if n == 0:
            return 0.0
        if n == 1:
            return self.ys[0]

        xs = np.array(self.xs)
        ys = np.array(self.ys)

        if n == 2:
            x0, x1 = xs
            y0, y1 = ys
            dx = x1 - x0
            if abs(dx) < self.epsilon:
                return float(0.5 * (y0 + y1))
            t = (x - x0) / dx
            return float(y0 + t * (y1 - y0))

        distances = np.abs(xs - x)
        nearest_idx = np.argsort(distances)[: self.k]
        xk = xs[nearest_idx]
        yk = ys[nearest_idx]
        weights = 1.0 / (distances[nearest_idx] + self.epsilon)

        slope, intercept = np.polyfit(xk, yk, 1, w=weights)
        return float(slope * x + intercept)
