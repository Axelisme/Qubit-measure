"""Candidate selection and isolated numerical spectrum fitting."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.failure import AnalysisFailed

from .base import fit_func, fitlor, fitsinc, lorfunc, sincfunc
from .peaks import PeakCandidate, detect_peaks
from .qubfreq import fit_qubit_freq


@dataclass(frozen=True)
class SpectrumFitOptions:
    """Numerical peak selection, independent of experiment forms.

    model_type selects a positive Lorentzian (lor) or signed sinc peak.
    expected_center_mhz selects the nearest candidate; None selects the most
    prominent. min_prominence is a fraction of the observed signal range,
    min_distance_mhz is candidate separation in MHz, min_snr is the minimum
    prominence and height above median in first-difference MAD noise units,
    and min_width_samples is the minimum observable half-prominence width in
    sample intervals (zero disables it). Detection thresholds must be finite
    and nonnegative; min_prominence must be at most one. fit_spectrum's
    local_peak mode ignores min_prominence, min_distance_mhz and min_width_samples
    and selects the highest nearby candidate rather than the nearest.
    """

    model_type: Literal["lor", "sinc"] = "lor"
    expected_center_mhz: float | None = None
    min_prominence: float = 0.2
    min_snr: float = 5.0
    min_width_samples: float = 0.0
    min_distance_mhz: float = 0.0


@dataclass(frozen=True)
class SpectrumFit:
    """Fitted MHz center/FWHM/errors and the local fitted observations.

    freqs, observed and fitted share the actual fitted interval. By default it
    is bounded halfway to neighbouring candidate centers; local_peak mode also
    limits its radius to six half-widths and ensures at least six samples.
    peaks contains all candidates in descending prominence order, selected_peak is the sampled target (None in a peakless
    legacy whole-window fit). Candidate widths are not fitted linewidths.
    """

    freq: float
    freq_err: float
    fwhm: float
    fwhm_err: float
    freqs: NDArray[np.float64]
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64]
    peaks: tuple[PeakCandidate, ...]
    selected_peak: PeakCandidate | None


def fit_spectrum(
    freqs: NDArray[np.float64],
    observed: NDArray[np.float64],
    options: SpectrumFitOptions,
    *,
    peaks: tuple[PeakCandidate, ...] | None = None,
    isolate_peaks: bool = True,
    local_peak: bool = False,
) -> SpectrumFit:
    """Fit a selected spectral peak with a midpoint or local six-half-width window.

    Inputs are finite, matching projected float64 vectors with at least two
    strictly monotonic MHz coordinates; uneven or descending sampling is allowed.
    The signal is a positive peak for lor or a signed principal-axis projection
    for sinc. Inputs are not modified. Supplied peaks must have been detected on
    these inputs with the effective detection thresholds for the chosen mode,
    and sorted by descending prominence.
    local_peak=True disables relative prominence, separation and sample-width
    gates. With an expected center, select the highest candidate within
    max(two median frequency steps, candidate.width / 2) of that center; without
    one, select the most prominent candidate. SNR still gates both prominence
    and height. The window extends six half-widths on each side, clipped at
    neighbouring candidate midpoints, then expanded to include the six nearest
    samples. PeakCandidate.width is the full width at half prominence (FWHM
    for an isolated zero-background peak), not HWHM: the radius is 3 * width.
    Local lor uses a positive peak with baseline allowed below the window
    minimum; local sinc seeds its width from the candidate rather than span.
    Clipped peaks use their observable width. Fewer than six total input samples
    still fails. freqs in the result reports the actual fitted window.
    isolate_peaks=False retains the legacy whole-window fit for single-mode
    workflows and ignores local_peak. No candidate or fewer than six isolated
    samples in default mode raises AnalysisFailed. No-candidate reasons
    distinguish prominence/SNR rejection from observable widths below
    min_width_samples, listing centers in MHz and
    widths in sample intervals. Invalid numeric inputs and optimizer failures
    propagate.
    """
    expected = options.expected_center_mhz
    local_peak = local_peak and isolate_peaks
    thresholds = (
        replace(
            options, min_prominence=0.0, min_width_samples=0.0, min_distance_mhz=0.0
        )
        if local_peak
        else options
    )
    if peaks is None:
        peaks = tuple(
            detect_peaks(
                freqs,
                observed,
                min_prominence=thresholds.min_prominence,
                min_distance=thresholds.min_distance_mhz,
                min_snr=thresholds.min_snr,
                min_width_samples=thresholds.min_width_samples,
            )
        )
    selected = (
        min(peaks, key=lambda peak: abs(peak.center - expected))
        if peaks and expected is not None
        else next(iter(peaks), None)
    )
    if local_peak:
        selected, mask = _local_peak_window(freqs, peaks, options)
        freqs, observed = freqs[mask], observed[mask]
    elif isolate_peaks:
        if selected is None:
            raise AnalysisFailed(_no_candidate_message(freqs, observed, options))
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
    if isolate_peaks and options.model_type == "lor" and selected is not None:
        # A broad peak can occupy most of the isolated window. Its median is
        # then signal, not background; seed the detected positive peak explicitly.
        baseline = float(np.min(observed))
        initial = [
            baseline,
            0.0,
            selected.height - baseline,
            selected.center,
            selected.width / 2,
        ]
        if local_peak:
            # A six-HWHM window still contains Lorentzian tails. Its minimum
            # is above the true baseline; fitlor's baseline bound biases width.
            parameters, covariance = fit_func(
                freqs,
                observed,
                lorfunc,
                init_p=initial,
                bounds=(
                    [-np.inf, -np.inf, 0.0, float(freqs.min()), 0.0],
                    [selected.height, np.inf, np.inf, float(freqs.max()), np.inf],
                ),
                fixedparams=[None, 0.0, None, None, None],
            )
        else:
            parameters, covariance = fitlor(
                freqs,
                observed,
                fitparams=initial,
                fixedparams=[None, 0.0, None, None, None],
            )
        errors = np.sqrt(np.diag(covariance))
        frequency, error = parameters[3], errors[3]
        width, width_error = 2 * parameters[4], 2 * errors[4]
        fitted = lorfunc(freqs, *parameters)
    elif local_peak and selected is not None:
        # Neighbouring sinc sidelobes shorten the window. Seed the principal
        # lobe from its detected width, not one tenth of the cropped span.
        baseline = float((observed[0] + observed[-1]) / 2)
        parameters, covariance = fitsinc(
            freqs,
            observed,
            fitparams=[
                baseline,
                0.0,
                selected.height - baseline,
                selected.center,
                selected.width / 1.2067,
            ],
        )
        errors = np.sqrt(np.diag(covariance))
        frequency, error = parameters[3], errors[3]
        width, width_error = 1.2067 * parameters[4], 1.2067 * errors[4]
        fitted = sincfunc(freqs, *parameters)
    else:
        frequency, error, width, width_error, fitted, _ = fit_qubit_freq(
            freqs, observed, options.model_type
        )
    return SpectrumFit(
        float(frequency),
        float(error),
        float(width),
        float(width_error),
        freqs,
        observed,
        fitted,
        peaks,
        selected,
    )


def _local_peak_window(
    freqs: NDArray[np.float64],
    peaks: tuple[PeakCandidate, ...],
    options: SpectrumFitOptions,
) -> tuple[PeakCandidate, NDArray[np.bool_]]:
    expected = options.expected_center_mhz
    step = float(np.median(np.abs(np.diff(freqs))))
    nearby = [
        peak
        for peak in peaks
        if expected is None
        or abs(peak.center - expected) <= max(2 * step, peak.width / 2)
    ]
    if not nearby:
        raise AnalysisFailed(
            f"Selected candidates=0 (detected={len(peaks)}); "
            f"min_snr={options.min_snr:g}, min_prominence=0, "
            "min_width_samples=0, min_distance_mhz=0; "
            f"no candidate near expected_center_mhz={expected} passes SNR thresholds"
        )
    selected = (
        max(nearby, key=lambda peak: peak.height) if expected is not None else nearby[0]
    )
    lower = max(
        (
            (peak.center + selected.center) / 2
            for peak in peaks
            if peak.center < selected.center
        ),
        default=-np.inf,
    )
    upper = min(
        (
            (peak.center + selected.center) / 2
            for peak in peaks
            if peak.center > selected.center
        ),
        default=np.inf,
    )
    radius = 3 * selected.width
    mask = (freqs >= max(lower, selected.center - radius)) & (
        freqs <= min(upper, selected.center + radius)
    )
    if freqs.size < 6:
        raise AnalysisFailed("Fewer than six total samples for selected peak fit")
    nearest = np.argsort(np.abs(freqs - selected.center), kind="stable")[:6]
    mask[nearest] = True
    return selected, mask


def _no_candidate_message(
    freqs: NDArray[np.float64],
    observed: NDArray[np.float64],
    options: SpectrumFitOptions,
) -> str:
    # Detect on sample indices to report observable widths even on uneven grids.
    # Peak separation cannot reject every candidate; omit it for this diagnostic.
    narrow_peaks = detect_peaks(
        np.arange(freqs.size, dtype=np.float64),
        observed,
        min_prominence=options.min_prominence,
        min_snr=options.min_snr,
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
        f"min_width_samples={options.min_width_samples:g} samples: {details}"
    )
