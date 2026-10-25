"""Pure spectroscopy and Rabi fits used by the shared flux calibrations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import (
    FitQuality,
    PeakCandidate,
    compute_fit_quality,
    fit_rabi,
)
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.twotone.freq.analysis import FreqAnalyzeOptions, fit_spectrum
from zcu_lab.v2.twotone.freq.core import FreqResult, qubfreq_signal2real
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiResult, rabi_signal2real
from zcu_lab.workflows.common.analysis import fit_or_failed, validate_fit_trace


@dataclass(frozen=True)
class FrequencyFit:
    """Lorentzian frequency estimate and same-coordinate projected curves.

    frequency_mhz/error_mhz are the fitted center and its marginal standard
    error. width_mhz/width_error_mhz are the FWHM and its standard error.
    freqs/observed/fitted are float64 vectors for the selected fit interval.
    peaks holds all candidates; selected_peak is the sampled target or None.
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
    freqs: NDArray[np.float64]
    peaks: tuple[PeakCandidate, ...]
    selected_peak: PeakCandidate | None
    source: FreqResult


@dataclass(frozen=True)
class RabiFit:
    """Lowest-mean-absolute-residual Rabi candidate with native phase estimation.

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


def analyze_qubit_freq(
    result: FreqResult,
    options: FreqAnalyzeOptions | None = None,
    *,
    peaks: tuple[PeakCandidate, ...] | None = None,
    isolate_peaks: bool = True,
) -> FrequencyFit | Failed:
    """Fit the spectroscopy trace to estimate the qubit resonance and linewidth.

    result uses MHz coordinates and complex128 signals of matching shape.
    Reuse the native background-removal/magnitude projection, detect all peaks,
    then fit the nearest expected or strongest candidate between neighbours.
    peaks may supply precomputed candidates on that projection. isolate_peaks=False
    retains single-mode whole-window fitting. Return FWHM/errors and local curves.
    Do not mutate result or acquire, plot, save, or apply an acceptance threshold.
    Return Failed(reason) for nonfinite measured IQ or the expected fit
    failures documented by fit_or_failed. Structural/configuration errors and
    unexpected projection/fitter errors propagate; no samples are replaced.
    """
    failure = validate_fit_trace(result.freqs, result.signals)
    if failure is not None:
        return failure
    options = options if options is not None else FreqAnalyzeOptions()
    observed = qubfreq_signal2real(result.signals)
    fit = fit_or_failed(
        lambda: fit_spectrum(
            result.freqs, observed, options, peaks=peaks, isolate_peaks=isolate_peaks
        )
    )
    if isinstance(fit, Failed):
        return fit
    quality = compute_fit_quality(fit.observed, fit.fitted, {}, None)
    return FrequencyFit(
        fit.freq,
        fit.freq_err,
        fit.fwhm,
        fit.fwhm_err,
        fit.observed,
        fit.fitted,
        quality,
        fit.freqs,
        fit.peaks,
        fit.selected_peak,
        result,
    )


def analyze_len_rabi(result: LenRabiResult) -> RabiFit | Failed:
    """Fit Rabi oscillations to estimate phase-aware pi and pi/2 durations.

    result uses microsecond coordinates and complex128 signals of matching
    1-D shape. Reuse the native real-axis rotation and phase-aware pi estimate.
    Estimating phase avoids a fixed-phase optimizer local minimum on native
    traces whose first sample is nonzero. Fit both decay and non-decay candidates,
    then use the lower mean absolute residual; one failed candidate is allowed.
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
    candidates = []
    for decay in (True, False):
        candidate = fit_or_failed(
            lambda decay=decay: fit_rabi(
                result.lengths, observed, decay=decay, init_phase=None
            )
        )
        if not isinstance(candidate, Failed):
            candidates.append(candidate)
    if not candidates:
        return Failed("Rabi decay and non-decay fits both failed")
    fit = min(
        candidates,
        key=lambda candidate: (
            float(np.mean(np.abs(observed - candidate[6])))
            if np.all(np.isfinite(candidate[6]))
            else math.inf
        ),
    )
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
