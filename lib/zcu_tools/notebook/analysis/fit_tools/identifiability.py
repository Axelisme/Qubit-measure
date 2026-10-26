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
    covariance: Log-parameter covariance, scaled by reduced_chi2.
    correlation: Unitless correlation, normalized before covariance scaling.
    rank: Numerical rank of the Jacobian, including zero columns.
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
    Reasons include the observed values. None returns an empty tuple.
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
                )
    return tuple(WeakParameter(name, tuple(why)) for name, why in reasons.items())


def calculate_fit_correlation(
    jacobian: NDArray[np.float64],
    cost: float,
    observation_count: float,
    parameter_names: tuple[str, ...],
) -> FitCorrelation | None:
    """Compute log-space covariance using the pseudoinverse of J.T @ J.

    Supply the residual Jacobian with respect to natural-log free parameters,
    the optimizer cost, effective observation count, and ordered free names.
    Return None for no free parameters or failed matrix decomposition. Invalid
    shapes or nonfinite inputs raise ValueError. Singular geometry retains its
    pseudoinverse covariance, rank and infinite normalized condition number.
    """
    count = len(parameter_names)
    jac = np.asarray(jacobian, dtype=np.float64)
    if jac.ndim != 2 or jac.shape[1] != count:
        raise ValueError("Jacobian columns must match free parameter names")
    if not np.all(np.isfinite(jac)) or not np.isfinite(cost) or cost < 0:
        raise ValueError("Jacobian and nonnegative cost must be finite")
    if not np.isfinite(observation_count) or observation_count <= 0:
        raise ValueError("observation_count must be positive and finite")
    if count == 0:
        return None
    try:
        unscaled = np.asarray(np.linalg.pinv(jac.T @ jac), dtype=np.float64)
        rank = int(np.linalg.matrix_rank(jac))
        norms = np.linalg.norm(jac, axis=0)
        condition = (
            float(np.linalg.cond(jac / norms))
            if rank == count and np.all(norms > 0)
            else np.inf
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
    return FitCorrelation(
        parameter_names, unscaled * scale, correlation, rank, condition, scale
    )
