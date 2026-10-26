"""Log-parameter uncertainty diagnostics shared by curve fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .loss import reduced_chi2_from_cost

CORRELATION_THRESHOLD = 0.95
LOG_STD_THRESHOLD = 1.0


@dataclass(frozen=True, slots=True)
class FitCorrelation:
    """Local geometry of the fitted natural-log parameters.

    parameter_names: Free parameter names, in matrix row/column order.
    covariance: Log-parameter covariance, scaled by reduced_chi2. Diagonal
        entries are inf for parameters with a component in the discarded space.
    correlation: Unitless correlation, normalized before covariance scaling;
        unconstrained parameters have NaN rows and columns.
    rank: Numerical rank of the column-normalized Jacobian.
    condition_number: Condition number after unit-norm column scaling; inf
        for rank-deficient geometry or a zero column.
    reduced_chi2: Optimizer objective times two divided by effective degrees
        of freedom (at least one).
    """

    parameter_names: tuple[str, ...]
    covariance: NDArray[np.float64]
    correlation: NDArray[np.float64]
    rank: int
    condition_number: float
    reduced_chi2: float


@dataclass(frozen=True, slots=True)
class WeakParameter:
    """A parameter name and all covariance/correlation reasons for flagging it."""

    parameter: str
    reasons: tuple[str, ...]


def find_weak_parameters(
    correlation: FitCorrelation | None,
) -> tuple[WeakParameter, ...]:
    """Flag |r| > CORRELATION_THRESHOLD or sigma(ln p) > LOG_STD_THRESHOLD.

    Return each weak parameter with all reasons, in matrix order. Log-space
    standard deviations are square roots of the scaled covariance diagonal.
    Reasons include the observed values; infinite uncertainty states that
    the data do not constrain the parameter. None returns an empty tuple.
    """
    reasons: dict[str, list[str]] = {}
    if correlation is not None:
        for row, name in enumerate(correlation.parameter_names):
            for col, other in enumerate(correlation.parameter_names):
                value = float(correlation.correlation[row, col])
                if row != col and abs(value) > CORRELATION_THRESHOLD:
                    reasons.setdefault(name, []).append(
                        f"|correlation| with {other} = {abs(value)} > "
                        f"{CORRELATION_THRESHOLD}"
                    )
            log_std = float(np.sqrt(correlation.covariance[row, row]))
            if log_std > LOG_STD_THRESHOLD:
                reasons.setdefault(name, []).append(
                    f"sigma(ln {name}) = {log_std} > {LOG_STD_THRESHOLD}"
                    + ("; not constrained by the data" if np.isinf(log_std) else "")
                )
    return tuple(WeakParameter(name, tuple(why)) for name, why in reasons.items())


def calculate_fit_correlation(
    jacobian: NDArray[np.float64],
    cost: float,
    observation_count: float,
    parameter_names: tuple[str, ...],
) -> FitCorrelation | None:
    """Compute log covariance and correlation from a residual Jacobian.

    Supply a 2D residual Jacobian with respect to natural-log free parameters,
    nonnegative optimizer cost, positive finite effective observation count,
    and ordered free names matching the Jacobian columns. Invalid shapes,
    negative cost or invalid observation count raise ValueError. Return None
    for nonfinite Jacobian/cost, no free parameters or failed decomposition.

    SVD uses unit-norm columns and relative tolerance eps * max(J.shape) for
    rank and discarded-direction membership. Covariance is transformed back
    to natural-log coordinates and scaled by optimizer-cost reduced chi2.
    Parameters with a discarded-direction component have infinite variance,
    even at zero cost, and NaN correlation rows/columns. Tiny independent
    columns retain finite, potentially large variance. Rank-deficient
    geometry has infinite condition number.
    """
    count = len(parameter_names)
    jac = np.asarray(jacobian, dtype=np.float64)
    if jac.ndim != 2 or jac.shape[1] != count:
        raise ValueError("Jacobian columns must match free parameter names")
    if not np.all(np.isfinite(jac)) or not np.isfinite(cost):
        return None
    if cost < 0:
        raise ValueError("cost must be nonnegative")
    if not np.isfinite(observation_count) or observation_count <= 0:
        raise ValueError("observation_count must be positive and finite")
    if count == 0:
        return None
    try:
        norms = np.linalg.norm(jac, axis=0)
        column_scales = np.where(norms > 0, norms, 1.0)
        normalized = jac / column_scales
        _, singular_values, directions = np.linalg.svd(
            normalized, full_matrices=jac.shape[0] < count
        )
        # Work on J, not J.T @ J, to avoid squaring its condition number.
        tolerance = np.finfo(np.float64).eps * max(jac.shape)
        cutoff = tolerance * singular_values[0] if singular_values.size else 0.0
        rank = int(np.count_nonzero(singular_values > cutoff))
        inverse_directions = (
            directions[:rank].T / singular_values[:rank] / column_scales[:, None]
        )
        unscaled = inverse_directions @ inverse_directions.T
        condition = (
            float(singular_values[0] / singular_values[-1]) if rank == count else np.inf
        )
    except np.linalg.LinAlgError:
        return None
    std = np.sqrt(np.maximum(np.diag(unscaled), 0))
    denominator = np.outer(std, std)
    correlation = np.divide(
        unscaled,
        denominator,
        out=np.full_like(unscaled, np.nan),
        where=denominator > 0,
    )
    scale = reduced_chi2_from_cost(cost, observation_count, count)
    # Null-space components mean the individual parameter has no finite variance.
    unconstrained = np.linalg.norm(directions[rank:], axis=0) > tolerance
    correlation[unconstrained, :] = np.nan
    correlation[:, unconstrained] = np.nan
    covariance = unscaled * scale
    covariance[np.diag_indices(count)] = np.where(
        unconstrained, np.inf, np.diag(covariance)
    )
    return FitCorrelation(
        parameter_names, covariance, correlation, rank, condition, scale
    )
