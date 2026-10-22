"""common analysis helpers for lab workflows."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_decay,
    fit_qubit_freq,
    fit_rabi,
)
from zcu_tools.experiment import AnalysisFailed
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.twotone.freq.core import FreqResult, qubfreq_signal2real
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiResult, rabi_signal2real
from zcu_lab.v2.twotone.time_domain.t1.core import T1Result, t1_signal2real


@dataclass(frozen=True)
class FrequencyFit:
    """Lorentzian frequency estimate and same-coordinate projected curves.

    frequency_mhz/error_mhz are the fitted center and its marginal standard
    error. width_mhz/width_error_mhz are the FWHM and its standard error.
    observed/fitted are float64 vectors aligned with source.freqs.
    quality contains residual diagnostics with no parameter-error entries.
    Values are not acceptance-filtered: native nonfinite errors and unavailable
    diagnostics remain visible for the workflow to reject.
    """

    frequency_mhz: float
    error_mhz: float
    width_mhz: float
    width_error_mhz: float
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    quality: FitQuality


@dataclass(frozen=True)
class RabiFit:
    """Nondecaying Rabi fit with native phase estimation.

    pi_length_us/error_us are the fitted pi-pulse duration and its propagated
    standard error. pi2_length_us/pi2_error_us retain the phase-aware native
    pi/2 duration and propagated error, not half the fitted pi duration.
    Both durations use the same acquisition waveform and gain, in microseconds.
    observed/fitted are float64 vectors aligned with
    source.lengths. quality contains residual diagnostics with no
    parameter-error entries. Native nonfinite estimates/errors are retained;
    this value does not certify calibration acceptance.
    """

    pi_length_us: float
    error_us: float
    pi2_length_us: float
    pi2_error_us: float
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    quality: FitQuality


@dataclass(frozen=True)
class DecayFit:
    """Single-exponential decay estimate without an acceptance decision.

    lifetime_us/error_us are the decay constant and its marginal standard
    error. observed/fitted are float64 vectors aligned with source.times for
    T1, or source.times[1:] for T2echo (the native first-sample exclusion).
    quality contains residual diagnostics with no parameter-error entries.
    Native nonfinite values or unavailable diagnostics remain visible.
    """

    lifetime_us: float
    error_us: float
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    quality: FitQuality


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


def analyze_qubit_freq(result: FreqResult) -> FrequencyFit | Failed:
    """Fit the spectroscopy trace to estimate the qubit resonance and linewidth.

    result uses MHz coordinates and complex128 signals of matching shape.
    Reuse the native background-removal/magnitude projection without skipping
    or masking observations. Return estimate, error, FWHM and residual curves.
    Do not mutate result or acquire, plot, save, or apply an acceptance threshold.
    Return Failed(reason) for nonfinite measured IQ or the expected fit
    failures documented by fit_or_failed. Structural/configuration errors and
    unexpected projection/fitter errors propagate; no samples are replaced.
    """
    failure = validate_fit_trace(result.freqs, result.signals)
    if failure is not None:
        return failure
    # Remove the native background, then fit the complete resonance trace.
    observed = qubfreq_signal2real(result.signals)
    fit = fit_or_failed(lambda: fit_qubit_freq(result.freqs, observed, type="lor"))
    if isinstance(fit, Failed):
        return fit
    frequency, error, width, width_error, fitted, _optimizer = fit
    # The workflow uses public estimates/errors, not positional optimizer vectors.
    quality = compute_fit_quality(observed, fitted, {}, None)
    return FrequencyFit(
        float(frequency),
        float(error),
        float(width),
        float(width_error),
        observed,
        fitted,
        quality,
    )


def analyze_len_rabi(result: LenRabiResult) -> RabiFit | Failed:
    """Fit Rabi oscillations to estimate phase-aware pi and pi/2 durations.

    result uses microsecond coordinates and complex128 signals of matching
    1-D shape. Reuse the native real-axis rotation and phase-aware pi estimate.
    Estimating phase avoids a fixed-phase optimizer local minimum on native
    traces whose first sample is nonzero.
    Return pi and pi/2 durations, propagated uncertainties and residual curves.
    Do not skip/mask data, mutate result or acquire, plot, save, or accept a fit.
    Nonfinite measured IQ and fit_or_failed's expected fit failures return
    Failed(reason). Structural/configuration and unexpected errors propagate.
    """
    failure = validate_fit_trace(result.lengths, result.signals)
    if failure is not None:
        return failure
    # Rotate IQ to the signal axis, then fit oscillations with free phase.
    observed = rabi_signal2real(result.signals)
    fit = fit_or_failed(
        lambda: fit_rabi(result.lengths, observed, decay=False, init_phase=None)
    )
    if isinstance(fit, Failed):
        return fit
    pi_length, error, pi2, pi2_error, _freq, _freq_error, fitted, _optimizer = fit
    quality = compute_fit_quality(observed, fitted, {}, None)
    return RabiFit(
        float(pi_length),
        float(error),
        float(pi2),
        float(pi2_error),
        observed,
        fitted,
        quality,
    )


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
    lifetime, error, fitted, _optimizer = fit
    # Preserve the native estimate and error; residual diagnostics do not refit.
    quality = compute_fit_quality(observed, fitted, {}, None)
    return DecayFit(float(lifetime), float(error), observed, fitted, quality)
