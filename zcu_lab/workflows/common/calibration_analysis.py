"""Pure spectroscopy and Rabi fits used by the shared flux calibrations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import (
    FitQuality,
    PeakCandidate,
    compute_fit_quality,
    detect_peaks,
    fit_qubit_freq,
    fit_rabi,
    fitlor,
    lorfunc,
)
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.twotone.freq.core import FreqResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiResult
from zcu_lab.workflows.common.analysis import fit_or_failed, validate_fit_trace
from zcu_lab.workflows.common.calibration_acquire import (
    qubfreq_signal2real,
    rabi_signal2real,
)
from zcu_lab.workflows.common.calibration_tunables import CalibrationTunables


@dataclass(frozen=True)
class FrequencyFit:
    """Lorentzian frequency estimate and same-coordinate projected curves.

    frequency_mhz/error_mhz are the fitted center and its marginal standard
    error. width_mhz/width_error_mhz are the FWHM and its standard error.
    freqs/observed/fitted are float64 vectors for the selected fit interval.
    peaks holds all candidates; selected_peak is the sampled target or None.
    source retains the full acquired spectrum for window gates and plotting.
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
    tun: CalibrationTunables,
    *,
    expected_center_mhz: float | None = None,
    peaks: tuple[PeakCandidate, ...] | None = None,
    isolate_peaks: bool = True,
) -> FrequencyFit | Failed:
    """Fit a Lorentzian to native MHz coordinates and complex128 IQ.

    tun supplies peak prominence, separation, SNR and observable width limits.
    Select the candidate nearest expected_center_mhz, or the strongest if None.
    peaks may reuse candidates detected with those limits. isolate_peaks fits
    between neighbouring peak midpoints; False fits the whole window.
    Return MHz center/FWHM/errors and same-coordinate projected fit curves.
    Nonfinite IQ, no isolated candidate, fewer than six isolated samples and
    expected fitter failures return Failed(reason). Invalid structure and
    unexpected errors propagate. Do not mutate, acquire, plot, save or accept.
    """
    failure = validate_fit_trace(result.freqs, result.signals)
    if failure is not None:
        return failure
    return fit_or_failed(
        lambda: _fit_frequency(
            result, tun, expected_center_mhz, peaks, isolate_peaks=isolate_peaks
        )
    )


def _fit_frequency(
    result: FreqResult,
    tun: CalibrationTunables,
    expected_center_mhz: float | None,
    peaks: tuple[PeakCandidate, ...] | None,
    *,
    isolate_peaks: bool,
) -> FrequencyFit:
    freqs = result.freqs
    observed = qubfreq_signal2real(result.signals)
    if peaks is None:
        peaks = tuple(
            detect_peaks(
                freqs,
                observed,
                min_prominence=tun.peak_min_prominence,
                min_distance=tun.peak_min_distance_mhz,
                min_snr=tun.peak_min_snr,
                min_width_samples=tun.peak_min_width_samples,
            )
        )
    selected = (
        min(peaks, key=lambda peak: abs(peak.center - expected_center_mhz))
        if peaks and expected_center_mhz is not None
        else next(iter(peaks), None)
    )
    if isolate_peaks:
        if selected is None:
            raise AnalysisFailed(_no_candidate_message(freqs, observed, tun))
        lower = max(
            (peak.center for peak in peaks if peak.center < selected.center),
            default=-np.inf,
        )
        upper = min(
            (peak.center for peak in peaks if peak.center > selected.center),
            default=np.inf,
        )
        mask = (freqs >= (lower + selected.center) / 2) & (
            freqs <= (upper + selected.center) / 2
        )
        freqs, observed = freqs[mask], observed[mask]
        if freqs.size < 6:
            raise AnalysisFailed("Fewer than six samples around selected peak")
    if isolate_peaks and selected is not None:
        # A broad peak can occupy most of the isolated window. Its median is
        # then signal, not background; seed the detected positive peak explicitly.
        baseline = float(np.min(observed))
        parameters, covariance = fitlor(
            freqs,
            observed,
            fitparams=[
                baseline,
                0.0,
                selected.height - baseline,
                selected.center,
                selected.width / 2,
            ],
            fixedparams=[None, 0.0, None, None, None],
        )
        errors = np.sqrt(np.diag(covariance))
        frequency, error = parameters[3], errors[3]
        width, width_error = 2 * parameters[4], 2 * errors[4]
        fitted = lorfunc(freqs, *parameters)
    else:
        frequency, error, width, width_error, fitted, _ = fit_qubit_freq(
            freqs, observed, "lor"
        )
    return FrequencyFit(
        float(frequency),
        float(error),
        float(width),
        float(width_error),
        observed,
        fitted,
        compute_fit_quality(observed, fitted, {}, None),
        freqs,
        peaks,
        selected,
        result,
    )


def _no_candidate_message(
    freqs: NDArray[np.float64],
    observed: NDArray[np.float64],
    tun: CalibrationTunables,
) -> str:
    # Detect on sample indices to report observable widths even on uneven grids.
    # Peak separation cannot reject every candidate; omit it for this diagnostic.
    narrow_peaks = detect_peaks(
        np.arange(freqs.size, dtype=np.float64),
        observed,
        min_prominence=tun.peak_min_prominence,
        min_snr=tun.peak_min_snr,
        min_width_samples=0,
    )
    if not narrow_peaks:
        return "No candidate peaks pass the prominence and SNR thresholds"
    details = "; ".join(
        f"{freqs[int(peak.center)]:.6g} MHz (observable width {peak.width:.3g} samples)"
        for peak in narrow_peaks
    )
    return (
        "Candidate peaks pass the prominence and SNR thresholds but fail "
        f"min_width_samples={tun.peak_min_width_samples:g} samples: {details}"
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
            float(np.mean(np.abs(observed - candidate.curve)))
            if np.all(np.isfinite(candidate.curve))
            else math.inf
        ),
    )
    pi_length, error = fit.pi_x, fit.pi_x_err
    pi2, pi2_error, fitted = fit.pi2_x, fit.pi2_x_err, fit.curve
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
