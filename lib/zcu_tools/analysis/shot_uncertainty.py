"""Shot-sampling uncertainty from retained IQ moments or class counts."""

import numpy as np
from numpy.typing import NDArray


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
    (NaN). Small negative variances from rounding are clipped at zero.
    Wrong moment shape or nonfinite projection weights raise ValueError.
    """
    if moments.shape[-1:] != (6,):
        raise ValueError("IQ moments must have shape (..., 6)")
    a, b = direction
    if not np.isfinite(direction).all():
        raise ValueError("IQ projection weights must be finite")

    n, i, q, ii, iq, qq = np.moveaxis(moments, -1, 0)
    n = np.where(n >= 2, n, np.nan)
    total = a * i + b * q
    sum_squared = a**2 * ii + 2 * a * b * iq + b**2 * qq
    centered_sum_squared = sum_squared - total**2 / n
    return np.asarray(np.sqrt(np.maximum(centered_sum_squared, 0) / (n * (n - 1))))


def class_probability_covariance(
    counts: NDArray[np.float64] | NDArray[np.int64],
) -> NDArray[np.float64]:
    """Estimate multinomial probability covariance from retained class counts.

    ``counts`` has shape ``(..., K)`` with K >= 2 mutually exclusive classes
    on the last axis. Each entry is a nonnegative count from retained shots only;
    a NaN marks a missing count. The result has shape ``(..., K, K)`` in squared
    probability units. It uses ``(diag(p) - p*p.T) / N`` per point, with ``N`` the
    sum of class counts and ``p = counts / N``. The square roots of the diagonal
    give each class's binomial SEM, including when there are more than two classes.
    A zero total or missing count gives a NaN matrix for that point. An observed
    fraction of zero or one gives zero plug-in variance, not a confidence bound.
    Invalid shape, negative or infinite counts raise ValueError.
    """
    if counts.ndim == 0 or counts.shape[-1] < 2:
        raise ValueError("class counts must have shape (..., K) with K >= 2")
    if np.any(counts < 0) or np.isinf(counts).any():
        raise ValueError("class counts must be nonnegative and not infinite")

    n = np.sum(counts, axis=-1, dtype=np.float64)
    n = np.where(n > 0, n, np.nan)
    p = counts / n[..., None]
    column = p[..., :, None]
    return column * (np.eye(counts.shape[-1]) - p[..., None, :]) / n[..., None, None]
