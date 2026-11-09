"""Detailed-balance generator, spectral propagator and analytic derivatives.

Population row vectors evolve from the left. For pair (i,j), gamma is the
sum of its two directed rates, and G[i,j] = gamma*pi[j]/(pi[i]+pi[j]). The
similarity S = sqrt(Pi) G inv(sqrt(Pi)) is symmetric. Working in S avoids
ill-conditioned nonsymmetric eigenvectors and gives the Daleckii–Krein
matrix-exponential derivative, including its repeated-eigenvalue limit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import exprel

from .model import Array

PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class Spectrum:
    """Symmetric spectrum and directions of S.

    pi/root: Stationary probabilities and their square roots, shape (3,).
    rate_directions/thermal_directions: dS/dgamma and dS/dpi, shape (3,3,3),
        direction first; pi entries are treated independently here.
    values/vectors: Eigenvalues descending (stationary, slow, fast) and their
        orthonormal eigenvectors as columns.
    """

    pi: Array
    root: Array
    rate_directions: Array
    thermal_directions: Array
    values: Array
    vectors: Array

    @property
    def ratio(self) -> Array:
        return self.root[None, :] / self.root[:, None]

    @property
    def ratio_log_gradient(self) -> Array:
        eye = np.eye(3)
        return (eye[None, :, :] - eye[:, None, :]) * (0.5 / self.pi)[None, None, :]


def spectrum(rates: Array, pi: Array) -> Spectrum:
    """Return S's spectrum and closed-form directions for positive pi."""
    matrix = np.zeros((3, 3))
    d_rate = np.zeros((3, 3, 3))
    d_pi = np.zeros((3, 3, 3))
    for p, (i, j) in enumerate(PAIRS):
        total = pi[i] + pi[j]
        basis = np.zeros((3, 3))
        basis[i, j] = basis[j, i] = np.sqrt(pi[i] * pi[j]) / total
        basis[i, i], basis[j, j] = -pi[j] / total, -pi[i] / total
        d_rate[p] = basis
        matrix += rates[p] * basis
        for a, b in ((i, j), (j, i)):
            step = np.zeros((3, 3))
            step[i, j] = step[j, i] = (
                np.sqrt(pi[b] / pi[a]) * (pi[b] - pi[a]) / (2 * total**2)
            )
            step[a, a], step[b, b] = pi[b] / total**2, -pi[b] / total**2
            d_pi[a] += rates[p] * step
    values, vectors = np.linalg.eigh(matrix)
    return Spectrum(pi, np.sqrt(pi), d_rate, d_pi, values[::-1], vectors[:, ::-1])


def _exponential_direction(spec: Spectrum, delays: Array, directions: Array) -> Array:
    values, vectors = spec.values, spec.vectors
    gap = np.abs(values[:, None] - values[None, :])
    top = np.maximum(values[:, None], values[None, :])
    x = delays[:, None, None]
    # This divided difference is x*exp(x*lambda) when the eigenvalues coincide.
    phi = x * np.exp(x * top) * exprel(-x * gap)
    conjugated = vectors.T @ directions @ vectors
    return np.moveaxis(vectors @ (conjugated * phi[:, None]) @ vectors.T, 1, -1)


def propagator_jacobian(spec: Spectrum, delays: Array) -> tuple[Array, Array, Array]:
    """Return exp(Gx), d/dgamma and d/dpi; directions are the final axis."""
    growth = np.exp(np.outer(delays, spec.values))
    symmetric = (spec.vectors * growth[:, None, :]) @ spec.vectors.T
    ratio = spec.ratio[None, :, :, None]
    d_rate = _exponential_direction(spec, delays, spec.rate_directions) * ratio
    d_pi = (
        _exponential_direction(spec, delays, spec.thermal_directions)
        + symmetric[:, :, :, None] * spec.ratio_log_gradient[None]
    ) * ratio
    return symmetric * spec.ratio[None], d_rate, d_pi


def mode_amplitude(spec: Spectrum, d0: Array, mode: int) -> float:
    """Thermal-weighted L1 population change carried by one spectral mode."""
    vector = spec.vectors[:, mode]
    projector = np.outer(vector, vector) * spec.ratio
    return float(spec.pi @ np.abs(d0 @ projector).sum(axis=1))
