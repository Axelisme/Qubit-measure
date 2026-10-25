"""coherence_fluxdep analysis helpers for lab workflows."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    compute_fit_quality,
    decay_time_bounds,
    fit_decay,
    fit_decay_fringe,
)
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


@dataclass(frozen=True)
class RamseyFit(DecayFit):
    """Ramsey decay fit with physical detune_mhz after removing the phase ramp.

    Inherited fields retain DecayFit's units and trace alignment.
    detune_mhz is fitted fringe frequency minus the actual artificial detune.
    """

    detune_mhz: float


def _fluxdep_projection(
    signals: NDArray[np.complex128], *, echo: bool
) -> NDArray[np.float64]:
    real = t2echo_signal2real(signals) if echo else t2ramsey_signal2real(signals)
    lower, upper = np.min(real), np.max(real)
    normalized = (real - lower) / (upper - lower + 1e-12)
    return 1.0 - normalized if real[0] < 0.5 * (upper + lower) else normalized


def analyze_t2echo_fluxdep(
    result: T2EchoResult, *, fit_fringe: bool
) -> DecayFit | Failed:
    """Fit Hahn-echo decay to estimate the refocused coherence lifetime.

    Supply matching 1-D microsecond times and complex IQ, at least five samples,
    with strictly increasing finite times. Normalize IQ to [0,1], reversing
    polarity when the initial signal is low. Pass fit_fringe=False for requested
    detune_ratio=0, True otherwise, even if the hardware rounds the ramp to zero.
    This selects fit_decay or fit_decay_fringe respectively. Both keep the
    first sample. Return DecayFit with
    whole-trace residual curves and microsecond estimates. Unlike the native
    CLI/overnight analysis, this fluxdep fit keeps sample zero. Invalid input
    raises ValueError for grid/shape errors. Nonfinite measured IQ and
    fit_or_failed's expected failures return Failed(reason); unexpected errors
    propagate. No acceptance or I/O occurs.
    """
    failure = _validate_fluxdep_trace(result.times, result.signals)
    if failure is not None:
        return failure
    # Normalize the complete trace with the flux-sweep polarity policy.
    observed = _fluxdep_projection(result.signals, echo=True)
    if not fit_fringe:
        fit = fit_or_failed(lambda: fit_decay(result.times, observed))
        if isinstance(fit, Failed):
            return fit
        lifetime, error, fitted, (parameters, covariance) = fit
        names = ("y0", "yscale", "decay_time")
    else:
        fringe = fit_or_failed(lambda: fit_decay_fringe(result.times, observed))
        if isinstance(fringe, Failed):
            return fringe
        lifetime, error, _detune, _detune_error, fitted, (parameters, covariance) = (
            fringe
        )
        names = ("y0", "yscale", "freq", "phase", "decay_time")
    return DecayFit(
        float(lifetime),
        float(error),
        observed,
        fitted,
        compute_fit_quality(
            observed,
            fitted,
            dict(zip(names, map(float, parameters), strict=True)),
            covariance,
            decay_bounds=decay_time_bounds(result.times),
        ),
        result.times,
    )


def analyze_t2ramsey(result: T2RamseyResult) -> RamseyFit | Failed:
    """Fit Ramsey fringes to estimate the unrefocused dephasing lifetime.

    Supply matching 1-D microsecond delays and complex IQ, at least five samples,
    with strictly increasing finite delays and finite runtime artificial-detune
    metadata. Missing metadata raises ValueError rather than reporting the
    artificial fringe frequency as physical detune. Native rotated IQ is normalized
    to [0,1] and initial-low polarity reversed. fit_decay_fringe estimates the
    microsecond lifetime and marginal error. detune_mhz subtracts the actual
    artificial phase-ramp frequency from the fitted fringe frequency.
    Observed/fitted retain all samples.
    Grid/shape errors raise ValueError. Nonfinite measured IQ and fit_or_failed's
    expected failures return Failed(reason); unexpected errors propagate. This
    function does not reject estimates, refit calibration, acquire, plot or save.
    """
    if result.true_activate_detune is None or not np.isfinite(
        result.true_activate_detune
    ):
        raise ValueError("Ramsey fluxdep requires actual artificial-detune metadata")
    failure = _validate_fluxdep_trace(result.times, result.signals)
    if failure is not None:
        return failure
    # Normalize the complete trace before fitting its decaying fringes.
    observed = _fluxdep_projection(result.signals, echo=False)
    fit = fit_or_failed(lambda: fit_decay_fringe(result.times, observed))
    if isinstance(fit, Failed):
        return fit
    lifetime, error, detune, _error, fitted, (parameters, covariance) = fit
    return RamseyFit(
        float(lifetime),
        float(error),
        observed,
        fitted,
        compute_fit_quality(
            observed,
            fitted,
            dict(
                zip(
                    ("y0", "yscale", "freq", "phase", "decay_time"),
                    map(float, parameters),
                    strict=True,
                )
            ),
            covariance,
            decay_bounds=decay_time_bounds(result.times),
        ),
        result.times,
        float(detune - result.true_activate_detune),
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
