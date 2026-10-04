"""Model-independent fit diagnostics from the optimizer's actual observations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QualityIssue:
    path: str
    reason: str


@dataclass(frozen=True)
class FitQuality:
    r2: float | None
    normalized_residual_rms: float | None
    relative_parameter_errors: Mapping[str, float | None]
    invalid: tuple[QualityIssue, ...]

    def to_summary_dict(self) -> dict[str, object]:
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
    """Use matching nonempty real vectors and covariance in parameter insertion order.

    Caller shape/name mistakes raise ValueError. Known unestimable metrics are
    None with their direct reason; missing covariance does not discard residual
    metrics. This helper does not refit or decide whether calibration is valid.
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
    if not np.isfinite(parameter):
        return None, "non_finite"
    if parameter == 0:
        return None, "zero_parameter"
    if variance is None:
        return None, "covariance_unavailable"
    if not np.isfinite(variance):
        return None, "non_finite"
    if variance < 0:
        return None, "negative_variance"
    return _ratio(float(np.sqrt(variance)), abs(parameter), "zero_parameter")
