"""Whitening and the independent chi-square block evidence null.

No data files or hardware state are read. The deterministic null table is built
on first use and retained in memory for subsequent live-curve updates.
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.linalg import solve_triangular
from scipy.special import lambertw, ndtri_exp
from scipy.stats import chi2


def whiten(
    means: NDArray[np.float64],
    counts: NDArray[np.float64],
    products: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return sqrt(N) L^-1(mean - componentwise median), shape (F,11).

    products pools feature outer products; subtract each point mean before
    estimating per-dwell covariance with sum(N)-F degrees of freedom.
    Raise ValueError when that covariance is not positive definite.
    """
    covariance = (products - means.T @ (counts[:, None] * means)) / (
        counts.sum() - len(counts)
    )
    if not np.isfinite(covariance).all():
        raise ValueError("within-point covariance must be finite and positive definite")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("within-point covariance must be symmetric positive definite")
    try:
        lower = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("within-point covariance must be positive definite") from exc
    centered = means - np.median(means, axis=0)
    return solve_triangular(lower, centered.T, lower=True).T * np.sqrt(counts[:, None])


def block_distances(
    whitened: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Squared distances of the first 2 and remaining 9 whitened features."""
    return (whitened[:, :2] ** 2).sum(axis=1), (whitened[:, 2:] ** 2).sum(axis=1)


def _evidence(distance: NDArray[np.float64], degrees: int) -> NDArray[np.float64]:
    scaled = np.maximum(distance / degrees, 1.0)
    return degrees * (scaled - 1 - np.log(scaled)) / 2


def _inverse_evidence(evidence: float, degrees: int) -> float:
    """Upper root D >= k of e(D,k)=evidence, using Lambert W branch -1."""
    if evidence <= 0:
        return float(degrees)
    return float(-degrees * lambertw(-np.exp(-1 - 2 * evidence / degrees), -1).real)


def _null_tail(evidence: float) -> float:
    """Continuous right-limit tail, including the exact B>threshold tail."""
    if evidence == 0:
        return float(1 - chi2.cdf(2, 2) * chi2.cdf(9, 9))
    upper = _inverse_evidence(evidence, 9)
    below_nine = chi2.cdf(9, 9) * np.exp(-_inverse_evidence(evidence, 2) / 2)

    def integrand(b: float) -> float:
        residual = evidence - (b - 9 - 9 * np.log(b / 9)) / 2
        return float(chi2.pdf(b, 9) * np.exp(-_inverse_evidence(residual, 2) / 2))

    middle, _ = quad(integrand, 9, upper, epsabs=0, epsrel=1e-8)
    return float(below_nine + middle + chi2.sf(upper, 9))


@lru_cache(maxsize=1)
def _null_table() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    grid = np.linspace(0, 80, 801)
    log_tail = np.log(np.array([_null_tail(float(t)) for t in grid]))
    grid.flags.writeable = False
    log_tail.flags.writeable = False
    return grid, log_tail


def evidence_logp(
    mean_distance: NDArray[np.float64], cross_distance: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Log P(T>=observed) for independent chi-square 2/9 block distances.

    Interpolate the exact one-dimensional null integral in log probability.
    Beyond the table, use its final log-tail slope. T=0 includes its atom (p=1).
    """
    statistic = _evidence(mean_distance, 2) + _evidence(cross_distance, 9)
    grid, log_tail = _null_table()
    slope = (log_tail[-1] - log_tail[-2]) / (grid[-1] - grid[-2])
    interpolated = np.interp(statistic, grid, log_tail)
    logp = np.where(
        statistic <= grid[-1],
        interpolated,
        log_tail[-1] + slope * (statistic - grid[-1]),
    )
    return np.where(statistic == 0, 0.0, logp)


def normal_sigma(logp: NDArray[np.float64]) -> NDArray[np.float64]:
    """Two-sided normal equivalent, evaluated in log space even below 1e-300."""
    return -ndtri_exp(logp - np.log(2.0))
