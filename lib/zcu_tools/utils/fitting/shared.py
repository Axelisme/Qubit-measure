from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from iminuit import Minuit
from numpy.typing import NDArray

Model = Callable[..., NDArray[np.float64]]
Limit = tuple[float | None, float | None]


@dataclass(frozen=True)
class ParameterSpec:
    """One global parameter declaration for a named shared fit."""

    name: str
    initial: float
    fixed: bool = False
    limits: Limit = (None, None)


@dataclass(frozen=True)
class FitTrace:
    """One least-squares trace with explicit global parameter identities."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    model: Model
    parameter_names: tuple[str, ...]
    errors: float | NDArray[np.float64] = 1.0


@dataclass(frozen=True)
class FitDiagnostics:
    valid: bool
    edm: float
    covariance_accurate: bool
    reached_call_limit: bool
    hesse_failed: bool
    reduced_chi_square: float


@dataclass(frozen=True)
class SharedFitResult:
    parameter_names: tuple[str, ...]
    values: Mapping[str, float]
    covariance: NDArray[np.float64]
    correlation: NDArray[np.float64]
    diagnostics: FitDiagnostics
    profile_intervals: Mapping[str, tuple[float, float]]

    def variance(self, name: str) -> float:
        return self.projected_variance({name: 1.0})

    def projected_variance(self, coefficients: Mapping[str, float]) -> float:
        """Return ``c.T @ covariance @ c`` for a named linear projection."""

        unknown = set(coefficients) - set(self.parameter_names)
        if unknown:
            raise KeyError(sorted(unknown)[0])
        vector = np.array(
            [coefficients.get(name, 0.0) for name in self.parameter_names],
            dtype=np.float64,
        )
        return float(vector @ self.covariance @ vector)


def _validate_parameters(
    parameters: Sequence[ParameterSpec], traces: Sequence[FitTrace]
) -> tuple[tuple[str, ...], dict[str, int]]:
    if not parameters:
        raise ValueError("at least one parameter is required")
    names = tuple(parameter.name for parameter in parameters)
    if any(not name for name in names):
        raise ValueError("parameter names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("each global parameter must be declared exactly once")

    indices = {name: index for index, name in enumerate(names)}
    for parameter in parameters:
        if not np.isfinite(parameter.initial):
            raise ValueError(f"initial value for {parameter.name!r} must be finite")
        lower, upper = parameter.limits
        if any(
            bound is not None and not np.isfinite(bound) for bound in (lower, upper)
        ):
            raise ValueError(f"limits for {parameter.name!r} must be finite or None")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"parameter {parameter.name!r} has empty limits")
        if lower is not None and parameter.initial < lower:
            raise ValueError(f"initial value for {parameter.name!r} is below its limit")
        if upper is not None and parameter.initial > upper:
            raise ValueError(f"initial value for {parameter.name!r} is above its limit")

    if not traces:
        raise ValueError("at least one fit trace is required")
    for trace in traces:
        if not trace.parameter_names:
            raise ValueError("each fit trace must name at least one parameter")
        if len(set(trace.parameter_names)) != len(trace.parameter_names):
            raise ValueError("a fit trace must not repeat a parameter name")
        unknown = set(trace.parameter_names) - indices.keys()
        if unknown:
            raise ValueError(
                f"fit trace references unknown parameters: {sorted(unknown)}"
            )
    return names, indices


def _trace_errors(trace: FitTrace) -> NDArray[np.float64]:
    errors = np.asarray(trace.errors, dtype=np.float64)
    if errors.ndim == 0:
        errors = np.full(trace.y.shape, float(errors), dtype=np.float64)
    if errors.shape != trace.y.shape:
        raise ValueError("trace errors must be scalar or match y shape")
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0.0):
        raise ValueError("trace errors must be positive and finite")
    return errors


def _validate_trace(trace: FitTrace) -> NDArray[np.float64]:
    if trace.x.ndim != 1:
        raise ValueError("fit trace x must be one-dimensional")
    if trace.x.size == 0 or trace.y.size == 0:
        raise ValueError("fit trace x and y must be non-empty")
    if trace.y.shape[0] != trace.x.size:
        raise ValueError("fit trace y leading dimension must match x")
    if np.any(~np.isfinite(trace.x)) or np.any(~np.isfinite(trace.y)):
        raise ValueError("fit trace data must be finite")
    return _trace_errors(trace)


def _covariance_arrays(
    fit: Minuit,
    names: tuple[str, ...],
    parameters: Sequence[ParameterSpec],
    scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    size = len(names)
    covariance = np.full((size, size), np.nan, dtype=np.float64)
    if fit.covariance is not None:
        covariance = np.asarray(fit.covariance, dtype=np.float64) * scale
    fixed = np.array([parameter.fixed for parameter in parameters], dtype=bool)
    covariance[fixed, :] = 0.0
    covariance[:, fixed] = 0.0

    correlation = np.full_like(covariance, np.nan)
    variances = np.diag(covariance)
    positive = variances > 0.0
    denominator = np.sqrt(np.outer(variances, variances))
    np.divide(covariance, denominator, out=correlation, where=denominator > 0.0)
    correlation[np.diag_indices(size)] = np.where(positive, 1.0, 0.0)
    correlation[fixed, :] = 0.0
    correlation[:, fixed] = 0.0
    return covariance, correlation


def fit_shared(
    traces: Sequence[FitTrace],
    parameters: Sequence[ParameterSpec],
    *,
    profile: Sequence[str] = (),
    max_calls: int | None = None,
) -> SharedFitResult:
    """Fit named least-squares traces with one global parameter declaration.

    Covariance is scaled by reduced chi-square, matching the existing
    ``curve_fit(absolute_sigma=False)`` fitting convention.
    """

    names, indices = _validate_parameters(parameters, traces)
    trace_errors = [_validate_trace(trace) for trace in traces]
    unknown_profiles = set(profile) - indices.keys()
    if unknown_profiles:
        raise ValueError(
            f"profile references unknown parameters: {sorted(unknown_profiles)}"
        )
    if max_calls is not None and max_calls <= 0:
        raise ValueError("max_calls must be positive")

    trace_indices = [
        np.array([indices[name] for name in trace.parameter_names], dtype=np.intp)
        for trace in traces
    ]

    def objective(*args: float) -> float:
        values = np.asarray(args, dtype=np.float64)
        total = 0.0
        for trace, errors, selected in zip(
            traces, trace_errors, trace_indices, strict=True
        ):
            predicted = np.asarray(
                trace.model(trace.x, *values[selected].tolist()), dtype=np.float64
            )
            if predicted.shape != trace.y.shape or np.any(~np.isfinite(predicted)):
                return np.inf
            residuals = ((trace.y - predicted) / errors).ravel()
            total += float(np.dot(residuals, residuals))
        return total

    fit = Minuit(
        objective,
        *(parameter.initial for parameter in parameters),
        name=names,
    )
    fit.errordef = Minuit.LEAST_SQUARES
    for parameter in parameters:
        fit.fixed[parameter.name] = parameter.fixed
        fit.limits[parameter.name] = parameter.limits
    fit.migrad(ncall=max_calls)
    fit.hesse()
    fval = fit.fval
    fmin = fit.fmin
    if fval is None or fmin is None:
        raise RuntimeError("iminuit did not return a fit minimum")

    free_count = sum(not parameter.fixed for parameter in parameters)
    degrees_of_freedom = sum(trace.y.size for trace in traces) - free_count
    reduced_chi_square = (
        float(fval) / degrees_of_freedom if degrees_of_freedom > 0 else np.nan
    )
    covariance_scale = reduced_chi_square if np.isfinite(reduced_chi_square) else 1.0
    covariance, correlation = _covariance_arrays(
        fit, names, parameters, covariance_scale
    )

    profile_intervals = {name: (np.nan, np.nan) for name in profile}
    if fmin.is_valid:
        for name in profile:
            if fit.fixed[name]:
                continue
            try:
                fit.minos(name)
            except RuntimeError:
                profile_intervals[name] = (np.nan, np.nan)
                continue
            error = fit.merrors[name]
            profile_intervals[name] = (
                float(fit.values[name] + error.lower) if error.lower_valid else np.nan,
                float(fit.values[name] + error.upper) if error.upper_valid else np.nan,
            )

    diagnostics = FitDiagnostics(
        valid=bool(fmin.is_valid),
        edm=float(fmin.edm),
        covariance_accurate=bool(fmin.has_accurate_covar),
        reached_call_limit=bool(fmin.has_reached_call_limit),
        hesse_failed=bool(fmin.hesse_failed),
        reduced_chi_square=reduced_chi_square,
    )
    values = {name: float(fit.values[name]) for name in names}
    return SharedFitResult(
        parameter_names=names,
        values=values,
        covariance=covariance,
        correlation=correlation,
        diagnostics=diagnostics,
        profile_intervals=profile_intervals,
    )
