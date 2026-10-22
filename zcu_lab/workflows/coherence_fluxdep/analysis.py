"""coherence_fluxdep analysis helpers for lab workflows."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import compute_fit_quality, fit_decay, fit_decay_fringe
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoResult, t2echo_signal2real
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyResult,
    t2ramsey_signal2real,
)
from zcu_lab.workflows.common.analysis import (
    DecayFit,
    fit_or_failed,
    validate_fit_trace,
)


def _fluxdep_projection(
    signals: NDArray[np.complex128], *, echo: bool
) -> NDArray[np.float64]:
    real = t2echo_signal2real(signals) if echo else t2ramsey_signal2real(signals)
    lower, upper = np.min(real), np.max(real)
    normalized = (real - lower) / (upper - lower + 1e-12)
    return 1.0 - normalized if real[0] < 0.5 * (upper + lower) else normalized


def analyze_t2echo_fluxdep(result: T2EchoResult) -> DecayFit | Failed:
    """Fit Hahn-echo decay to estimate the refocused coherence lifetime.

    Supply matching 1-D microsecond times and complex IQ, at least five samples,
    with strictly increasing finite times and explicit zero detune. Normalize
    native rotated IQ to [0,1], reversing polarity when the initial signal is
    low, then fit_decay without skipping the first sample. Return DecayFit with
    whole-trace residual curves and microsecond estimates. Unlike the native
    CLI/overnight analysis, this fluxdep fit keeps sample zero. Invalid input
    raises ValueError for grid/shape/detune errors. Nonfinite measured IQ and
    fit_or_failed's expected failures return Failed(reason); unexpected errors
    propagate. No acceptance or I/O occurs.
    """
    if result.true_activate_detune != 0.0:
        raise ValueError("Echo fluxdep requires explicit zero detune")
    failure = _validate_fluxdep_trace(result.times, result.signals)
    if failure is not None:
        return failure
    # Normalize the complete trace with the flux-sweep polarity policy.
    observed = _fluxdep_projection(result.signals, echo=True)
    fit = fit_or_failed(lambda: fit_decay(result.times, observed))
    if isinstance(fit, Failed):
        return fit
    lifetime, error, fitted, _ = fit
    return DecayFit(
        float(lifetime),
        float(error),
        observed,
        fitted,
        compute_fit_quality(observed, fitted, {}, None),
    )


def analyze_t2ramsey(result: T2RamseyResult) -> DecayFit | Failed:
    """Fit Ramsey fringes to estimate the unrefocused dephasing lifetime.

    Supply matching 1-D microsecond delays and complex IQ, at least five samples,
    with strictly increasing finite delays. Native rotated IQ is normalized
    to [0,1] and initial-low polarity reversed. fit_decay_fringe estimates the
    microsecond lifetime and marginal error. Observed/fitted retain all samples.
    Grid/shape errors raise ValueError. Nonfinite measured IQ and fit_or_failed's
    expected failures return Failed(reason); unexpected errors propagate. This
    function does not reject estimates, refit calibration, acquire, plot or save.
    """
    failure = _validate_fluxdep_trace(result.times, result.signals)
    if failure is not None:
        return failure
    # Normalize the complete trace before fitting its decaying fringes.
    observed = _fluxdep_projection(result.signals, echo=False)
    fit = fit_or_failed(lambda: fit_decay_fringe(result.times, observed))
    if isinstance(fit, Failed):
        return fit
    lifetime, error, _detune, _error, fitted, _ = fit
    return DecayFit(
        float(lifetime),
        float(error),
        observed,
        fitted,
        compute_fit_quality(observed, fitted, {}, None),
    )


def _validate_fluxdep_trace(
    times: NDArray[np.float64], signals: NDArray[np.complex128]
) -> Failed | None:
    if (
        times.ndim != 1
        or signals.ndim != 1
        or times.shape != signals.shape
        or len(times) < 5
    ):
        raise ValueError(
            "T2 fluxdep requires matching 1-D traces with at least five samples"
        )
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("T2 fluxdep requires increasing finite delays")
    return validate_fit_trace(times, signals)
