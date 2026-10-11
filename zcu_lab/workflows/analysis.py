"""Pure workflow fits; acquisition, plotting and acceptance have other owners.

Native projections and fitting helpers own the numerical models. These results
surface their public estimates and uncertainties, not positional optimizer
vectors. FitQuality supplies residual diagnostics only: its parameter mapping
is empty. Calibration policy must not interpret a residual metric as readout SNR.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    compute_fit_quality,
    fit_decay,
    fit_decay_fringe,
    fit_qubit_freq,
    fit_rabi,
)

from zcu_lab.v2.twotone.freq.core import FreqResult, qubfreq_signal2real
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiResult, rabi_signal2real
from zcu_lab.v2.twotone.time_domain.t1.core import T1Result, t1_signal2real
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoResult, t2echo_signal2real
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyResult,
    t2ramsey_signal2real,
)


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


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project MIST IQ with the old centred-magnitude/spread policy, no fit.

    signals is one complex 1-D gain trace. Return same-shaped absolute centred
    magnitude divided by its population standard deviation. All-NaN traces
    stay NaN; zero spread retains native division diagnostics rather than a
    threshold, epsilon or invented success. Input is unchanged; no I/O occurs.
    """
    if signals.ndim != 1:
        raise ValueError("MIST projection requires a 1-D gain trace")
    if np.all(np.isnan(signals)):
        return np.abs(signals)
    magnitude = np.abs(signals - np.mean(signals))
    return magnitude / np.std(magnitude)


def analyze_qubit_freq(result: FreqResult) -> FrequencyFit:
    """Fit the complete canonical 1-D frequency trace with a Lorentzian.

    result uses MHz coordinates and complex128 signals of matching shape.
    Reuse the native background-removal/magnitude projection without skipping
    or masking observations. Return estimate, error, FWHM and residual curves.
    Do not mutate result or acquire, plot, save, or apply an acceptance threshold.
    Native projection/fitting errors and optimizer warnings propagate; native
    fallback values remain distinguishable through errors/quality.
    """
    observed = qubfreq_signal2real(result.signals)
    frequency, error, width, width_error, fitted, _optimizer = fit_qubit_freq(
        result.freqs, observed, type="lor"
    )
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


def analyze_len_rabi(result: LenRabiResult) -> RabiFit:
    """Fit the complete canonical length-Rabi trace, with free phase and no decay.

    result uses microsecond coordinates and complex128 signals of matching
    1-D shape. Reuse the native real-axis rotation and phase-aware pi estimate.
    Estimating phase avoids a fixed-phase optimizer local minimum on native
    traces whose first sample is nonzero.
    Return pi and pi/2 durations, propagated uncertainties and residual curves.
    Do not skip/mask data, mutate result or acquire, plot, save, or accept a fit.
    Native projection/fitting errors and optimizer warnings propagate.
    """
    observed = rabi_signal2real(result.signals)
    pi_length, error, pi2, pi2_error, _freq, _freq_error, fitted, _optimizer = fit_rabi(
        result.lengths, observed, decay=False, init_phase=None
    )
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


def analyze_t1(result: T1Result) -> DecayFit:
    """Fit the complete canonical T1 trace with a single exponential.

    result uses microsecond coordinates and complex128 signals of matching
    1-D shape. Reuse the native real-axis rotation without skipping/masking.
    Return decay constant, marginal uncertainty and residual curves. Do not
    mutate result or acquire, plot, save, or apply an acceptance threshold.
    Native projection/fitting errors and optimizer warnings propagate.
    """
    observed = t1_signal2real(result.signals)
    lifetime, error, fitted, _optimizer = fit_decay(result.times, observed)
    # Preserve the native estimate and error; residual diagnostics do not refit.
    quality = compute_fit_quality(observed, fitted, {}, None)
    return DecayFit(float(lifetime), float(error), observed, fitted, quality)


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


def analyze_t2echo(result: T2EchoResult) -> DecayFit:
    """Fit a native zero-detune echo trace with a single exponential.

    Supply matching 1-D float64 times in microseconds and complex128 IQ.
    true_activate_detune must explicitly be 0.0, not None or a fringe value.
    Exclude the first sample before projection/fitting, as native analysis
    does. The remaining trace needs at least four finite samples with
    strictly increasing times; invalid inputs raise ValueError.
    Return lifetime, marginal uncertainty and residual curves aligned with
    result.times[1:]. Do not mutate result or acquire, plot, save, or apply
    an acceptance threshold. Native errors and optimizer warnings propagate;
    native nonfinite estimates and unavailable diagnostics remain visible.
    """
    if result.true_activate_detune != 0.0:
        raise ValueError("T2echo decay analysis requires explicit zero detune")
    if (
        result.times.ndim != 1
        or result.signals.ndim != 1
        or result.times.shape != result.signals.shape
    ):
        raise ValueError("T2echo times and signals must have matching 1-D shape")
    times, signals = result.times[1:], result.signals[1:]
    if len(times) < 4:
        raise ValueError("T2echo needs at least four samples after the first")
    if not np.isfinite(times).all() or not np.isfinite(signals).all():
        raise ValueError("T2echo retained samples must be finite")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("T2echo retained times must be strictly increasing")
    observed = t2echo_signal2real(signals)
    lifetime, error, fitted, _optimizer = fit_decay(times, observed)
    quality = compute_fit_quality(observed, fitted, {}, None)
    return DecayFit(float(lifetime), float(error), observed, fitted, quality)
