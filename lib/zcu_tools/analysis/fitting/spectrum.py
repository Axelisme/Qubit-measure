"""Candidate selection and isolated numerical spectrum fitting."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.failure import AnalysisFailed

from .base import fitlor, lorfunc
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
    and nonnegative; min_prominence must be at most one.
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

    freqs, observed and fitted share the selected interval, bounded halfway to
    neighbouring candidate centers. peaks contains all candidates in descending
    prominence order, selected_peak is the sampled target (None in a peakless
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
) -> SpectrumFit:
    """Fit the nearest expected or most prominent candidate, excluding neighbours.

    Inputs are finite, matching projected float64 vectors with at least two
    strictly monotonic MHz coordinates; uneven or descending sampling is allowed.
    The signal is a positive peak for lor or a signed principal-axis projection
    for sinc. Inputs are not modified. Supplied peaks must have been detected on
    these inputs with options' thresholds and sorted by descending prominence.
    isolate_peaks=False retains the legacy whole-window fit for single-mode
    workflows. No candidate or fewer than six isolated samples raises
    AnalysisFailed. No-candidate reasons distinguish prominence/SNR rejection
    from observable widths below min_width_samples, listing centers in MHz and
    widths in sample intervals. Invalid numeric inputs and optimizer failures
    propagate.
    """
    expected = options.expected_center_mhz
    if peaks is None:
        peaks = tuple(
            detect_peaks(
                freqs,
                observed,
                min_prominence=options.min_prominence,
                min_distance=options.min_distance_mhz,
                min_snr=options.min_snr,
                min_width_samples=options.min_width_samples,
            )
        )
    selected = (
        min(peaks, key=lambda peak: abs(peak.center - expected))
        if peaks and expected is not None
        else next(iter(peaks), None)
    )
    if isolate_peaks:
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
