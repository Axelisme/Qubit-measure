"""Shot-sampling uncertainty from retained IQ moments or class counts."""

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

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
    moments: NDArray[np.float64],
    *,
    direction: tuple[float | NDArray[np.float64], float | NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Estimate the standard error of the mean of ``a*I + b*Q`` per point.

    ``moments`` has shape ``(..., 6)`` in shot-sums order (ADR-0074):
    ``[N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]``. The sums include only the ``N`` retained
    shots at each point. ``direction=(a, b)`` gives finite I/Q weights. Each
    weight is a scalar or an array broadcastable to the point shape ``(...)``.
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
    a, b = (np.asarray(weight, dtype=np.float64) for weight in direction)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("IQ projection weights must be finite")
    try:
        a = np.broadcast_to(a, moments.shape[:-1])
        b = np.broadcast_to(b, moments.shape[:-1])
    except ValueError as error:
        raise ValueError(
            "IQ projection weights must broadcast to the point shape"
        ) from error

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


def propagated_iq_shot_sem(
    moments: NDArray[np.float64], *, gradients: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """Combine first-order shot SEM for independent IQ means at each point.

    ``moments`` has shape ``(..., groups, 6)`` with at least one group and
    the shot-sums order and validation of ``iq_shot_sem``. ``gradients`` has
    shape ``(..., groups)``; each complex entry holds the real observable's
    partial derivatives with respect to that group's mean I and Q. Evaluate
    these derivatives at the observed means. Preserve I/Q covariance within
    each group and add independent group variances. Return ``(...)`` in the
    observable's units. Missing group statistics leave the point's SEM NaN.
    Nonfinite gradients or propagated SEM leave NaN and log a warning.
    Mismatched shapes or invalid shot statistics raise ValueError.
    """
    if (
        moments.ndim < 2
        or moments.shape[-2] == 0
        or moments.shape[:-1] != gradients.shape
    ):
        raise ValueError("IQ gradients must match the point and group shape of moments")
    finite = np.isfinite(gradients)
    i = np.where(finite, gradients.real, 0.0)
    q = np.where(finite, gradients.imag, 0.0)
    # Normalize before projecting so a finite but large derivative is not squared.
    scale = np.maximum(np.abs(i), np.abs(q))
    divisor = np.where(scale == 0, 1.0, scale)
    unit_sem = iq_shot_sem(moments, direction=(i / divisor, q / divisor))
    with np.errstate(over="ignore"):
        sem = np.hypot.reduce(unit_sem * scale, axis=-1)
    unavailable = ~finite.all(axis=-1) | np.isinf(sem)
    if np.any(unavailable):
        logger.warning(
            "Nonfinite IQ gradients or propagated shot SEM at %d point(s)",
            np.count_nonzero(unavailable),
        )
    return np.where(unavailable, np.nan, sem)


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


def corrected_population_sem(
    counts: NDArray[np.float64] | NDArray[np.int64],
    confusion_matrix: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Return per-class shot SEMs, treating the readout calibration as fixed.

    ``counts`` follows ``class_probability_covariance``. ``confusion_matrix``
    is (K, K), with true-state rows and classified-state columns. Propagate
    the covariance through its transposed inverse; None leaves it unchanged.
    The result has the same shape as ``counts``. Singular or nonfinite
    calibration and other undefined SEMs produce NaN and a warning. A
    wrongly shaped matrix or invalid counts raise ValueError.
    """
    covariance = class_probability_covariance(counts)
    if confusion_matrix is None:
        sem = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    else:
        classes = counts.shape[-1]
        if confusion_matrix.shape != (classes, classes):
            raise ValueError("confusion matrix must match both class axes")
        if not np.isfinite(confusion_matrix).all():
            logger.warning("Omitting corrected shot SEMs: nonfinite confusion matrix")
            return np.full(counts.shape, np.nan)
        try:
            jacobian = np.linalg.solve(confusion_matrix.T, np.eye(classes))
        except np.linalg.LinAlgError:
            logger.warning("Omitting corrected shot SEMs: singular confusion matrix")
            return np.full(counts.shape, np.nan)
        # Each class pair contributes a rank-one covariance with weight -Σ[i, j]
        # along the difference of their one-hot vectors. Take the norm of the
        # projected square-root factors so a representable SEM need not be squared.
        i, j = np.triu_indices(classes, k=1)
        scale = np.sqrt(-covariance[..., None, i, j])
        left, right = jacobian[:, i], jacobian[:, j]
        with np.errstate(over="ignore", invalid="ignore"):
            difference = left - right
            projected = np.where(
                np.isfinite(difference),
                scale * difference,
                scale * left - scale * right,
            )
            sem = np.hypot.reduce(projected, axis=-1)
    finite = np.isfinite(sem)
    missing = finite.size - np.count_nonzero(finite)
    if missing:
        logger.warning("Omitting %d shot SEMs: nonfinite sampling variance", missing)
    return np.where(finite, sem, np.nan)
