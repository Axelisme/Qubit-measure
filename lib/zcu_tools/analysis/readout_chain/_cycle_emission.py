"""Gaussian emissions and analytic truncated-exponential tail moments.

A read that leaves state i for j gives IQ mean c_j + u*(c_i - c_j), where the
jump time u on [0,1] follows a truncated exponential, plus isotropic Gaussian
noise (see CycleNuisances). Confluent hypergeometric functions give E[u**k] in
closed form, without quadrature; derivatives use -Cov(u**k,u).
"""

from functools import lru_cache
from math import comb

import numpy as np
from scipy.special import hyp1f1

from ._cycle_features import ORDERS
from ._cycle_layout import (
    ROW_CENTERS,
    ROW_LANDING,
    ROW_RETENTION,
    ROW_SIZE,
    ROW_VARIANCE,
)


def simplex(s: float, f: float) -> np.ndarray:
    return np.array([s * f, s * (1 - f), 1 - s])


def simplex_coordinates(p: np.ndarray) -> np.ndarray:
    s = float(p[0] + p[1])
    return np.array([s, p[0] / s if s else 0.5])


def simplex_gradient(s: float, f: float) -> np.ndarray:
    return np.array([[f, s], [1 - f, -s], [-1.0, 0.0]])


def _polynomial(mean: float, delta: float, order: int, variance: float) -> np.ndarray:
    result = np.zeros(6)
    central = [1.0, 0.0, variance, 0.0, 3 * variance**2]
    for power in range(order + 1):
        scale = comb(order, power) * central[order - power]
        for k in range(power + 1):
            result[k] += scale * comb(power, k) * mean ** (power - k) * delta**k
    return result


def _product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(a, b)[:6]


def _tail(p: float) -> tuple[np.ndarray, np.ndarray]:
    if p >= 1:
        raise ValueError("deterministic read jumps have no regular finite Jacobian")
    rate = -np.log1p(-p)
    k = np.arange(6)
    moments = hyp1f1(k + 1, k + 2, -rate) / ((k + 1) * hyp1f1(1, 2, -rate))
    derivative = np.zeros(6)
    derivative[:5] = -(moments[1:] - moments[:5] * moments[1]) / (1 - p)
    return moments, derivative


def _edge(
    centers: np.ndarray, variance: float, i: int, j: int, p: float, max_order: int
) -> tuple[np.ndarray, np.ndarray]:
    delta = centers[i] - centers[j]
    u, du = _tail(p) if i != j else (np.array([1.0, 0, 0, 0, 0, 0]), np.zeros(6))
    polys = [
        [
            _polynomial(centers[j, a], delta[a], k, variance)
            for k in range(max_order + 1)
        ]
        for a in (0, 1)
    ]
    value, gradient = np.zeros(15), np.zeros((15, 8))
    for o, (a, b) in enumerate(ORDERS):
        if a + b > max_order:
            continue
        poly = _product(polys[0][a], polys[1][b])
        value[o] = poly @ u
        gradient[o, 7] = poly @ du
        for axis, power in enumerate((a, b)):
            if power == 0:
                continue
            reduced = _product(polys[0][a - (axis == 0)], polys[1][b - (axis == 1)])
            start = power * (reduced[:5] @ u[1:])
            total = power * (reduced @ u)
            gradient[o, axis] = start
            gradient[o, 2 + axis] = total - start
            if power >= 2:
                twice = _product(
                    polys[0][a - 2 * (axis == 0)], polys[1][b - 2 * (axis == 1)]
                )
                gradient[o, 6] += power * (power - 1) / 2 * (twice @ u)
    return value, gradient


def emissions(row: np.ndarray, max_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Emission matrices (15,3,3) and derivatives (15,3,3,14).

    Row: the shared-row layout of _cycle_layout (centers, variance, thermal
    s/f, retention, landing s/f).
    All dynamics derivatives are analytic, including tails at probability zero.
    """
    centers, variance = row[ROW_CENTERS].reshape(3, 2), row[ROW_VARIANCE]
    keep = row[ROW_RETENTION]
    rho = simplex(*row[ROW_LANDING])
    drho = simplex_gradient(*row[ROW_LANDING])
    transition = np.diag(keep) + (1 - keep[:, None]) * rho[None, :]
    matrices, gradients = np.zeros((15, 3, 3)), np.zeros((15, 3, 3, ROW_SIZE))
    for i in range(3):
        for j in range(3):
            p = float(transition[i, j])
            value, grad = _edge(centers, variance, i, j, p, max_order)
            matrices[:, i, j] = p * value
            gradients[:, i, j, 2 * i : 2 * i + 2] += p * grad[:, :2]
            gradients[:, i, j, 2 * j : 2 * j + 2] += p * grad[:, 2:4]
            gradients[:, i, j, ROW_VARIANCE] = p * grad[:, 6]
            dp = np.zeros(ROW_SIZE)
            dp[ROW_RETENTION.start + i] = (i == j) - rho[j]
            dp[ROW_LANDING] = (1 - keep[i]) * drho[j]
            gradients[:, i, j] += (value + p * grad[:, 7])[:, None] * dp
    return matrices, gradients


@lru_cache(maxsize=256)
def cached_emissions(
    row: tuple[float, ...], max_order: int
) -> tuple[np.ndarray, np.ndarray]:
    # Per-point projection changes drive probabilities, not the emission law.
    # Cached matrices stay private; batching below copies them into fresh arrays.
    return emissions(np.asarray(row), max_order)
