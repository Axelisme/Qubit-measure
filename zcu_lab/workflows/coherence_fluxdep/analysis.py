"""coherence_fluxdep analysis helpers for lab workflows."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import compute_fit_quality, fit_decay, fit_decay_fringe

from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoResult, t2echo_signal2real
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyResult,
    t2ramsey_signal2real,
)
from zcu_lab.workflows.common.analysis import DecayFit


def _fluxdep_projection(
    signals: NDArray[np.complex128], *, echo: bool
) -> NDArray[np.float64]:
    real = t2echo_signal2real(signals) if echo else t2ramsey_signal2real(signals)
    lower, upper = np.min(real), np.max(real)
    normalized = (real - lower) / (upper - lower + 1e-12)
    return 1.0 - normalized if real[0] < 0.5 * (upper + lower) else normalized


def analyze_t2echo_fluxdep(result: T2EchoResult) -> DecayFit:
    """Fit the complete finite zero-detune Echo trace with old fluxdep policy.

    Supply matching 1-D microsecond times and complex IQ, at least five finite
    samples, with strictly increasing times and explicit zero detune. Normalize
    native rotated IQ to [0,1], reversing polarity when the initial signal is
    low, then fit_decay without skipping the first sample. Return DecayFit with
    whole-trace residual curves and microsecond estimates. Unlike the native
    CLI/overnight analysis, this fluxdep fit keeps sample zero. Invalid input
    raises ValueError; optimizer errors propagate. No acceptance or I/O occurs.
    """
    if result.true_activate_detune != 0.0:
        raise ValueError("Echo fluxdep requires explicit zero detune")
    _validate_fluxdep_trace(result.times, result.signals)
    observed = _fluxdep_projection(result.signals, echo=True)
    lifetime, error, fitted, _ = fit_decay(result.times, observed)
    return DecayFit(
        float(lifetime),
        float(error),
        observed,
        fitted,
        compute_fit_quality(observed, fitted, {}, None),
    )


def analyze_t2ramsey(result: T2RamseyResult) -> DecayFit:
    """Fit a complete finite Ramsey fringe using the old fluxdep projection.

    Supply matching 1-D microsecond delays and complex IQ, at least five finite
    samples, with strictly increasing delays. Native rotated IQ is normalized
    to [0,1] and initial-low polarity reversed. fit_decay_fringe estimates the
    microsecond lifetime and marginal error. Observed/fitted retain all samples.
    Invalid inputs raise ValueError; optimizer errors propagate. This function
    does not reject estimates, refit calibration, acquire, plot or write files.
    """
    _validate_fluxdep_trace(result.times, result.signals)
    observed = _fluxdep_projection(result.signals, echo=False)
    lifetime, error, _detune, _error, fitted, _ = fit_decay_fringe(
        result.times, observed
    )
    return DecayFit(
        float(lifetime),
        float(error),
        observed,
        fitted,
        compute_fit_quality(observed, fitted, {}, None),
    )


def _validate_fluxdep_trace(
    times: NDArray[np.float64], signals: NDArray[np.complex128]
) -> None:
    if (
        times.ndim != 1
        or signals.ndim != 1
        or times.shape != signals.shape
        or len(times) < 5
    ):
        raise ValueError(
            "T2 fluxdep requires matching 1-D traces with at least five samples"
        )
    if (
        not np.isfinite(times).all()
        or not np.isfinite(signals).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError(
            "T2 fluxdep requires finite signals and increasing finite delays"
        )
