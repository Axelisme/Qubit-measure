"""Shot-sampling uncertainty from retained IQ moments or class counts."""

import numpy as np
from numpy.typing import NDArray

_ROUNDING_RTOL = 16 * np.finfo(np.float64).eps


def _validate_counts(counts: NDArray[np.float64] | NDArray[np.int64]) -> None:
    finite = counts[~np.isnan(counts)]
    if (
        np.any(finite < 0)
        or np.isinf(finite).any()
        or np.any(finite != np.trunc(finite))
    ):
        raise ValueError("shot counts must be finite nonnegative integers or NaN")


def _validate_centered_iq(
    n: NDArray[np.float64],
    i: NDArray[np.float64],
    q: NDArray[np.float64],
    ii: NDArray[np.float64],
    iq: NDArray[np.float64],
    qq: NDArray[np.float64],
) -> None:
    """Check both variances and the covariance before choosing a projection."""
    with np.errstate(over="ignore", invalid="ignore"):
        i_mean_term = i * (i / n)
        q_mean_term = q * (q / n)
        # Divide the larger factor first to avoid losing a tiny factor to underflow.
        cross_mean_term = np.where(np.abs(i) >= np.abs(q), (i / n) * q, i * (q / n))
        centered_i = ii - i_mean_term
        centered_q = qq - q_mean_term
        centered_cross = iq - cross_mean_term
    if np.isinf([centered_i, centered_q, centered_cross]).any():
        raise ValueError("centered IQ moments overflow float64")

    i_tol = _ROUNDING_RTOL * np.maximum(np.abs(ii), i_mean_term)
    q_tol = _ROUNDING_RTOL * np.maximum(np.abs(qq), q_mean_term)
    if np.any(centered_i < -i_tol) or np.any(centered_q < -q_tol):
        raise ValueError("IQ moments give a negative centered sum of squares")

    cross_tol = _ROUNDING_RTOL * np.maximum(np.abs(iq), np.abs(cross_mean_term))
    # Allow rounding in each difference, including when a variance rounds to zero.
    # Bound |cross| by the product of square roots, not the overflowing determinant.
    i_bound = np.hypot(np.sqrt(np.maximum(centered_i, 0)), np.sqrt(i_tol))
    q_bound = np.hypot(np.sqrt(np.maximum(centered_q, 0)), np.sqrt(q_tol))
    with np.errstate(over="ignore"):
        cross_bound = i_bound * q_bound
    if np.any(np.abs(centered_cross) - cross_bound > cross_tol):
        raise ValueError("centered IQ cross sum exceeds the covariance bound")


def iq_shot_sem(
    moments: NDArray[np.float64], *, direction: tuple[float, float]
) -> NDArray[np.float64]:
    """Estimate the standard error of the mean of ``a*I + b*Q`` per point.

    ``moments`` has shape ``(..., 6)`` in shot-sums order (ADR-0074):
    ``[N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]``. The sums include only the ``N`` retained
    shots at each point. ``direction=(a, b)`` gives finite linear weights.
    Holding those weights fixed, the result has shape ``(...)`` and the units
    of the projected observable. It is the sample standard deviation (ddof=1)
    divided by sqrt(N).
    Fewer than two retained shots or NaN moments leave the error undefined
    (NaN). A rounding-scale negative centered sum of squares is clipped to zero.
    Invalid shape, nonfinite weights, invalid counts, or infinite moments
    raise ValueError. At defined points, both centered sums of squares and the
    centered cross sum must describe a valid covariance up to rounding, even
    outside the projection direction. Invalid covariance or overflow raises
    ValueError.
    """
    if moments.shape[-1:] != (6,):
        raise ValueError("IQ moments must have shape (..., 6)")
    a, b = np.asarray(direction, dtype=np.float64)
    if not np.isfinite(direction).all():
        raise ValueError("IQ projection weights must be finite")

    n, i, q, ii, iq, qq = np.moveaxis(moments, -1, 0)
    _validate_counts(n)
    if np.isinf(moments).any():
        raise ValueError("IQ moments must not be infinite")
    defined = (n >= 2) & ~np.isnan(moments).any(axis=-1)
    n = np.where(defined, n, np.nan)
    _validate_centered_iq(n, i, q, ii, iq, qq)
    with np.errstate(over="ignore", invalid="ignore"):
        total = a * i + b * q
        i_term = a * (a * ii)
        iq_term = 2 * (a * (b * iq))
        q_term = b * (b * qq)
        mean_term = total * (total / n)
        centered = i_term + iq_term + q_term - mean_term
    if np.any(defined & ~np.isfinite(centered)):
        raise ValueError("IQ projection overflows finite moments")
    # Cancellation can expose rounding of any of the four subtracted terms.
    scale = np.maximum.reduce(
        [np.abs(i_term), np.abs(iq_term), np.abs(q_term), mean_term]
    )
    if np.any(centered < -_ROUNDING_RTOL * scale):
        raise ValueError("IQ moments give a negative centered sum of squares")
    return np.asarray(np.sqrt(np.maximum(centered, 0)) / np.sqrt(n) / np.sqrt(n - 1))


def class_probability_covariance(
    counts: NDArray[np.float64] | NDArray[np.int64],
) -> NDArray[np.float64]:
    """Estimate multinomial probability covariance from retained class counts.

    ``counts`` has shape ``(..., K)`` with K >= 2 mutually exclusive classes
    on the last axis. Each entry is a nonnegative count from retained shots only;
    a NaN marks a missing count. The result has shape ``(..., K, K)`` in squared
    probability units. It uses ``(diag(p) - p*p.T) / (N-1)`` per point, with
    ``N`` the sum of class counts and ``p = counts / N``. This is the one-hot
    sample covariance (ddof=1) divided by N. The square roots of the diagonal
    give each class's binomial SEM, including when there are more than two classes.
    Fewer than two retained shots or a missing count gives a NaN matrix for that
    point. For N >= 2, a fraction of zero or one gives zero sample variance, not
    a confidence bound. Invalid shape, negative, fractional or infinite counts,
    or a total that overflows float64, raise ValueError.
    """
    if counts.ndim == 0 or counts.shape[-1] < 2:
        raise ValueError("class counts must have shape (..., K) with K >= 2")
    _validate_counts(counts)

    with np.errstate(over="ignore"):
        n = np.sum(counts, axis=-1, dtype=np.float64)
    if np.isinf(n).any():
        raise ValueError("total shot count overflows float64")
    n = np.where(n >= 2, n, np.nan)
    p = counts / n[..., None]
    column = p[..., :, None]
    return (
        column * (np.eye(counts.shape[-1]) - p[..., None, :]) / (n - 1)[..., None, None]
    )
