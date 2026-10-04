"""Model-independent fit diagnostics from the optimizer's actual observations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QualityIssue:
    """The direct reason one field of a computed FitQuality is unestimable.

    Attributes:
        path: Relative to FitQuality or its summary: "r2",
            "normalized_residual_rms", or "relative_parameter_errors.<name>".
            After the last fixed prefix, <name> is the complete mapping key;
            dots in a parameter name do not denote further nested fields.
        reason: One of six codes: "zero_variance" (constant observations for r2),
            "zero_range" (constant observations for normalized residual RMS),
            "zero_parameter" (zero relative-error denominator),
            "covariance_unavailable" (no parameter covariance),
            "negative_variance" (negative covariance diagonal), or "non_finite"
            (non-finite inputs, intermediate arithmetic, or result).

    An issue explains a None, not an invalid fit or calibration.
    """

    path: str
    reason: str


@dataclass(frozen=True)
class FitQuality:
    """Model-independent diagnostics; no fit/calibration acceptance policy.

    Attributes:
        r2: 1 - sum((y - y_fit)**2) / sum((y - mean(y))**2), without clamping;
            negative values are retained. None for zero observation variance
            (zero_variance) or non-finite data/arithmetic (non_finite).
        normalized_residual_rms: sqrt(mean((y - y_fit)**2)) / (max(y) - min(y)).
            y is the actual optimizer observations after any skip/mask, not
            the full raw trace. None for zero observation range (zero_range)
            or non-finite data/arithmetic (non_finite).
        relative_parameter_errors: Each original parameter name maps to
            sqrt(covariance[i, i]) / abs(parameter), or None. Missing covariance
            keeps every name with None/covariance_unavailable. Otherwise the
            first applicable reason is non_finite parameter, non_finite
            diagonal, negative_variance, zero_parameter, then non_finite ratio.
            Off-diagonal entries do not affect these marginal errors.
        invalid: Direct explanations of unestimable fields, not fit validity.
            In computed results, every None has exactly one matching
            QualityIssue, and no finite field has an issue. Relative-error
            paths use the entire parameter name, even when it contains dots.

    compute_fit_quality returns finite numbers or None. This dataclass stores
    those results without validating manually constructed instances.
    """

    r2: float | None
    normalized_residual_rms: float | None
    relative_parameter_errors: Mapping[str, float | None]
    invalid: tuple[QualityIssue, ...]

    def to_summary_dict(self) -> dict[str, object]:
        """Copy a computed result into the fixed JSON-safe summary structure.

        Keys are r2, normalized_residual_rms, relative_parameter_errors (a dict
        retaining every name), and invalid (a list of {"path", "reason"} dicts).
        None remains None and serializes as JSON null. A result returned by
        compute_fit_quality supports json.dumps(..., allow_nan=False); manually
        constructed instances are not validated or repaired here.
        """
        return {
            "r2": self.r2,
            "normalized_residual_rms": self.normalized_residual_rms,
            "relative_parameter_errors": dict(self.relative_parameter_errors),
            "invalid": [
                {"path": issue.path, "reason": issue.reason} for issue in self.invalid
            ],
        }


def compute_fit_quality(
    y: NDArray[np.float64],
    y_fit: NDArray[np.float64],
    parameters: Mapping[str, float],
    covariance: NDArray[np.float64] | None,
) -> FitQuality:
    """Compute diagnostics from the actual optimizer inputs, without refitting.

    Args:
        y: Nonempty 1-D real observations passed to the optimizer after any
            skip/mask. Real integer and floating-point arrays are accepted.
        y_fit: Model values at exactly the same observations/coordinates as y,
            with the same shape and also a real numeric dtype.
        parameters: Parameter values keyed by nonblank string names. Mapping
            insertion order defines covariance row/column order. An empty
            mapping is allowed; dots in names remain part of the key.
        covariance: Real square matrix of shape (len(parameters), len(parameters))
            in that order, or None when unavailable. Only the diagonal is used
            for marginal errors; no symmetry or positive-semidefinite gate is
            applied to off-diagonal entries.

    Returns:
        FitQuality with the definitions and None/issue correspondence documented
        on that type. Missing covariance takes priority over parameter-value
        errors, keeps all parameter keys, and does not discard residual metrics.
        Non-finite data/arithmetic, zero denominators, and negative diagonal
        variances produce None plus a direct issue, rather than raising.

    Raises:
        ValueError: Observation/model vectors are empty, not 1-D, mismatched in
            shape, or complex; a parameter name is not a nonblank string; or
            covariance has the wrong shape or is complex.

    Inputs are not modified. No optimizer, calibration, accept, or writeback
    policy is changed.
    """
    _validate_observations(y, y_fit)
    _validate_parameter_names(parameters)
    if covariance is not None and covariance.shape != (
        len(parameters),
        len(parameters),
    ):
        raise ValueError("covariance shape must match the named parameter order")
    if covariance is not None and not np.isrealobj(covariance):
        raise ValueError("covariance must be real")

    r2, normalized_rms, issues = _residual_metrics(y, y_fit)
    relative_errors: dict[str, float | None] = {}
    for index, (name, value) in enumerate(parameters.items()):
        variance = None if covariance is None else float(covariance[index, index])
        error, reason = _relative_error(value, variance)
        relative_errors[name] = error
        if reason is not None:
            issues.append(QualityIssue(f"relative_parameter_errors.{name}", reason))
    return FitQuality(r2, normalized_rms, relative_errors, tuple(issues))


def _validate_parameter_names(names: Iterable[object]) -> None:
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Parameter names must be nonempty strings")


def _validate_observations(y: NDArray[np.float64], y_fit: NDArray[np.float64]) -> None:
    if y.ndim != 1 or y_fit.ndim != 1:
        raise ValueError("Observations and model values must be one-dimensional")
    if y.size == 0 or y_fit.size == 0:
        raise ValueError("Observations and model values must be nonempty")
    if y.shape != y_fit.shape:
        raise ValueError("Observations and model values must have matching shapes")
    if not np.isrealobj(y) or not np.isrealobj(y_fit):
        raise ValueError("Observations and model values must be real")


def _ratio(
    numerator: float, denominator: float, zero_reason: str
) -> tuple[float | None, str | None]:
    """Return a finite ratio/no reason, or None/one direct reason.

    Non-finite inputs/result use non_finite; a zero denominator uses zero_reason.
    """
    if not np.isfinite([numerator, denominator]).all():
        return None, "non_finite"
    if denominator == 0:
        return None, zero_reason
    value = numerator / denominator
    if not np.isfinite(value):
        return None, "non_finite"
    return float(value), None


def _residual_metrics(
    y: NDArray[np.float64], y_fit: NDArray[np.float64]
) -> tuple[float | None, float | None, list[QualityIssue]]:
    if not np.isfinite(y).all() or not np.isfinite(y_fit).all():
        return (
            None,
            None,
            [
                QualityIssue("r2", "non_finite"),
                QualityIssue("normalized_residual_rms", "non_finite"),
            ],
        )
    with np.errstate(over="ignore", invalid="ignore"):
        residual_squared = (y - y_fit) ** 2
        residual_sum = float(np.sum(residual_squared))
        variance_sum = float(np.sum((y - np.mean(y)) ** 2))
        rms = float(np.sqrt(np.mean(residual_squared)))
        span = float(np.ptp(y))
    fraction, r2_reason = _ratio(residual_sum, variance_sum, "zero_variance")
    normalized_rms, rms_reason = _ratio(rms, span, "zero_range")
    issues = [
        QualityIssue(path, reason)
        for path, reason in (("r2", r2_reason), ("normalized_residual_rms", rms_reason))
        if reason is not None
    ]
    return None if fraction is None else 1 - fraction, normalized_rms, issues


def _relative_error(
    parameter: float, variance: float | None
) -> tuple[float | None, str | None]:
    """Return a marginal relative error, with missing covariance taking priority."""
    if variance is None:
        return None, "covariance_unavailable"
    if not np.isfinite(parameter):
        return None, "non_finite"
    if not np.isfinite(variance):
        return None, "non_finite"
    if variance < 0:
        return None, "negative_variance"
    return _ratio(float(np.sqrt(variance)), abs(parameter), "zero_parameter")
