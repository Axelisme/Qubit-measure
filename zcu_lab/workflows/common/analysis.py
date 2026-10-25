"""common analysis helpers for lab workflows."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    decay_time_bounds,
    fit_decay,
)
from zcu_tools.experiment import AnalysisFailed
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.twotone.time_domain.t1.core import T1Result, t1_signal2real


@dataclass(frozen=True)
class DecayFit:
    """Single-exponential decay estimate without an acceptance decision.

    lifetime_us/error_us are the decay constant and its marginal standard
    error. observed/fitted are float64 vectors aligned with source.times for
    T1, or source.times[1:] for T2echo (the native first-sample exclusion).
    quality contains diagnostics from the actual parameters/covariance and
    decay bounds. times contains the microsecond delays passed to the fitter,
    aligned with observed/fitted, without resetting the delay origin.
    Native nonfinite values or unavailable diagnostics remain visible.
    """

    lifetime_us: float
    error_us: float
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    quality: FitQuality
    times: NDArray[np.float64]


def lifetime_rejection(
    quality: FitQuality | None,
    times: NDArray[np.float64],
    lifetime_us: float,
    min_lifetime_points: int,
) -> str | None:
    """Reject invalid diagnostics or too few acquired delays within one lifetime.

    quality is the actual fit diagnostics; None rejects unavailable quality.
    times contains only delays used by the fitter, in microseconds, with no
    origin shift. lifetime_us must be finite and positive or is rejected.
    Count 0 <= delay <= lifetime_us, including the endpoints. Fewer than
    min_lifetime_points (a positive integer, normally 5) rejects with the count.
    Invalid quality issues retain their paths/reasons. Return None only when
    both gates pass. This helper does not update history or change any fit.
    """
    if type(min_lifetime_points) is not int or min_lifetime_points < 1:
        raise ValueError("min_lifetime_points must be a positive integer")
    if not np.isfinite(lifetime_us) or lifetime_us <= 0:
        return "Lifetime must be finite and positive"
    if quality is None:
        return "Fit quality unavailable"
    reasons = [f"{issue.path}: {issue.reason}" for issue in quality.invalid]
    count = int(np.count_nonzero((times >= 0) & (times <= lifetime_us)))
    if count < min_lifetime_points:
        reasons.append(
            f"Only {count} delay points within first lifetime (0 <= delay <= "
            f"{lifetime_us:.9g} us); require at least {min_lifetime_points}"
        )
    return "; ".join(reasons) if reasons else None


def validate_fit_trace(
    coordinates: NDArray[np.float64], signals: NDArray[np.complex128]
) -> Failed | None:
    """Check a native 1-D trace before IQ projection, without changing samples.

    coordinates are the finite sweep coordinates in the caller's native units.
    Mismatched, empty or non-1-D arrays and nonfinite coordinates raise ValueError
    as structural/configuration errors. Nonfinite measured IQ returns Failed
    with a diagnostic; otherwise return None. Callers retain their own grid,
    sample-count and detune requirements. No fit or acceptance decision occurs.
    """
    if coordinates.ndim != 1 or signals.ndim != 1 or coordinates.shape != signals.shape:
        raise ValueError("Fit coordinates and signals must have matching 1-D shape")
    if coordinates.size == 0 or not np.isfinite(coordinates).all():
        raise ValueError("Fit coordinates must be nonempty and finite")
    if not np.isfinite(signals).all():
        return Failed("Fit failed: measured signals contain NaN or infinity")
    return None


def fit_or_failed[T](fit: Callable[[], T]) -> T | Failed:
    """Run one native fit, isolating only NaN data and non-convergence.

    fit is a zero-argument callable that invokes the caller's chosen fitter.
    Return its value unchanged on success. The fitter's AnalysisFailed and
    scipy's exact nonfinite-array ValueError become Failed(reason). Warnings
    retain their normal handling. Other exceptions, including callback errors
    and configuration/shape ValueErrors, propagate unchanged.
    This shared boundary serves spectroscopy, Rabi and lifetime analysis; it
    does not mask data, change optimizer options or impose acceptance gates.
    """
    try:
        return fit()
    except AnalysisFailed as exc:
        return Failed(f"Fit failed: {exc}")
    except ValueError as exc:
        if str(exc) != "array must not contain infs or NaNs":
            raise
        return Failed(f"Fit failed: {exc}")


def analyze_t1(result: T1Result) -> DecayFit | Failed:
    """Fit population decay to estimate the energy-relaxation lifetime.

    result uses microsecond coordinates and complex128 signals of matching
    1-D shape. Reuse the native real-axis rotation without skipping/masking.
    Return decay constant, marginal uncertainty and residual curves. Do not
    mutate result or acquire, plot, save, or apply an acceptance threshold.
    Nonfinite measured IQ and fit_or_failed's expected fit failures return
    Failed(reason). Structural/configuration and unexpected errors propagate.
    """
    failure = validate_fit_trace(result.times, result.signals)
    if failure is not None:
        return failure
    # Rotate IQ to the signal axis, then fit a single exponential.
    observed = t1_signal2real(result.signals)
    fit = fit_or_failed(lambda: fit_decay(result.times, observed))
    if isinstance(fit, Failed):
        return fit
    lifetime, error, fitted, (parameters, covariance) = fit
    quality = compute_fit_quality(
        observed,
        fitted,
        dict(zip(("y0", "yscale", "decay_time"), map(float, parameters), strict=True)),
        covariance,
        decay_bounds=decay_time_bounds(result.times),
    )
    return DecayFit(
        float(lifetime), float(error), observed, fitted, quality, result.times
    )
