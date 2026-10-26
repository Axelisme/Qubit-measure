"""Log-parameter uncertainty and profile diagnostics shared by curve fits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .loss import reduced_chi2_from_cost

CORRELATION_THRESHOLD = 0.95
PROFILE_DELTA_CHI2_THRESHOLD = 1.0


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
        of freedom (at least one); also the profile objective scale.
    """

    parameter_names: tuple[str, ...]
    covariance: NDArray[np.float64]
    correlation: NDArray[np.float64]
    rank: int
    condition_number: float
    reduced_chi2: float


@dataclass(frozen=True, slots=True)
class ParameterProfile:
    """One-dimensional profile in physical parameter units.

    parameter: Free parameter being scanned.
    values: Ascending positive scan grid, including best_value.
    delta_chi2: Twice the optimizer cost increase divided by the original
        covariance's reduced chi-squared, one entry per grid point.
    best_value: Original fitted value, in physical units.
    bounds: Original physical lower/upper fit bounds (upper may be inf).
    interval: Lower/upper crossings at delta chi-squared = 1; None if absent.
    lower_unbounded, upper_unbounded: True when that side has no crossing.
    lower_reason, upper_reason: None for a crossing, otherwise "bound" or
        "scan range", naming the edge reached without a crossing.
    """

    parameter: str
    values: NDArray[np.float64]
    delta_chi2: NDArray[np.float64]
    best_value: float
    bounds: tuple[float, float]
    interval: tuple[float | None, float | None]
    lower_unbounded: bool
    upper_unbounded: bool
    lower_reason: Literal["bound", "scan range"] | None
    upper_reason: Literal["bound", "scan range"] | None


@dataclass(frozen=True, slots=True)
class WeakParameter:
    """A parameter name and all correlation/profile reasons for flagging it."""

    parameter: str
    reasons: tuple[str, ...]


def find_weak_parameters(
    correlation: FitCorrelation | None,
    profiles: Sequence[ParameterProfile],
) -> tuple[WeakParameter, ...]:
    """Flag |r| > CORRELATION_THRESHOLD or either open profile side.

    Return one entry per weak parameter with all reasons. Correlation order
    comes first, followed by profile-only names. Missing matrices and omitted
    profiles provide no evidence and do not themselves imply weakness.
    """
    reasons: dict[str, list[str]] = {}
    if correlation is not None:
        for row, name in enumerate(correlation.parameter_names):
            for col, other in enumerate(correlation.parameter_names):
                value = float(correlation.correlation[row, col])
                if row != col and abs(value) > CORRELATION_THRESHOLD:
                    reasons.setdefault(name, []).append(
                        f"|correlation| with {other} = {abs(value):.3f} > "
                        f"{CORRELATION_THRESHOLD}"
                    )
    for profile in profiles:
        if profile.lower_unbounded:
            reasons.setdefault(profile.parameter, []).append(
                f"lower profile unbounded ({profile.lower_reason})"
            )
        if profile.upper_unbounded:
            reasons.setdefault(profile.parameter, []).append(
                f"upper profile unbounded ({profile.upper_reason})"
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
