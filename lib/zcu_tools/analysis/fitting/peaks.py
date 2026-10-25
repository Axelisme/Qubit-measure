"""Peak candidates in one-dimensional projected numeric signals."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_widths


@dataclass(frozen=True)
class PeakCandidate:
    """Sampled center, absolute signal height, half-prominence width and prominence.

    center/width use the input axis units; height/prominence use signal units.
    truncated means a half-prominence crossing lies outside the measured window.
    Its width is only the observable portion, not an extrapolated full width.
    snr is (height - signal median) / estimated noise sigma, not prominence/sigma.
    """

    center: float
    height: float
    width: float
    prominence: float
    truncated: bool = False
    snr: float = 0.0


def detect_peaks(
    detune: NDArray[np.float64],
    signal: NDArray[np.float64],
    *,
    min_prominence: float = 0.2,
    min_distance: float = 0.0,
    min_snr: float = 5.0,
    min_width_samples: float = 3.0,
) -> list[PeakCandidate]:
    """Find positive peaks in matching finite float64 vectors (at least 2 points).

    detune must be strictly monotonic; uneven/descending sampling is supported.
    min_prominence is in [0, 1], relative to signal peak-to-peak. min_distance is
    finite, nonnegative and uses axis units. min_snr (default 5) requires both
    prominence and height above the signal median to exceed min_snr * sigma.
    Estimate sigma from first-difference MAD / (0.67448975 * sqrt(2)), with a
    machine-epsilon * signal-range floor (at least float tiny) for zero MAD.
    min_width_samples (default 3) is the observable half-prominence width in
    fractional sample intervals, including on uneven axes. Both noise knobs
    must be finite and nonnegative; zero disables the corresponding gate.
    These statistical gates cannot exclude every possible noise realization.
    Keep the more prominent candidate
    when centers are closer than min_distance. Return descending prominence,
    with ascending center breaking ties. Flat signals return an empty list.
    Endpoints use the measured minimum as background; clipped half-prominence
    widths are flagged truncated. Inputs are not mutated. Invalid inputs raise
    ValueError. Negative-going signals must be inverted by the caller.
    """
    _validate_inputs(detune, signal, min_prominence, min_distance)
    if not np.isfinite(min_snr) or min_snr < 0:
        raise ValueError("min_snr must be finite and nonnegative")
    if not np.isfinite(min_width_samples) or min_width_samples < 0:
        raise ValueError("min_width_samples must be finite and nonnegative")
    differences = np.diff(signal)
    sigma = float(np.median(np.abs(differences - np.median(differences)))) / (
        0.6744897501960817 * np.sqrt(2)
    )
    # Keep SNR finite for noiseless/quantized traces, including a zero MAD.
    sigma = max(
        sigma, np.finfo(float).eps * float(np.ptp(signal)), np.finfo(float).tiny
    )
    baseline = float(np.median(signal))
    if detune[0] > detune[-1]:
        detune, signal = detune[::-1], signal[::-1]
    # A background sample beyond each edge makes clipped peaks observable.
    padded = np.pad(signal, 1, constant_values=float(np.min(signal)))
    indices, properties = find_peaks(
        padded, prominence=max(min_prominence * float(np.ptp(signal)), min_snr * sigma)
    )
    _, _, left, right = peak_widths(padded, indices, rel_height=0.5)
    samples = np.arange(signal.size)
    candidates = [
        PeakCandidate(
            float(detune[index - 1]),
            float(signal[index - 1]),
            float(
                np.interp(right_ip - 1, samples, detune)
                - np.interp(left_ip - 1, samples, detune)
            ),
            float(prominence),
            bool(left_ip < 1 or right_ip > signal.size),
            float((signal[index - 1] - baseline) / sigma),
        )
        for index, left_ip, right_ip, prominence in zip(
            indices, left, right, properties["prominences"], strict=True
        )
        if min(right_ip, float(signal.size)) - max(left_ip, 1.0) >= min_width_samples
    ]
    selected: list[PeakCandidate] = []
    for peak in sorted(candidates, key=lambda peak: -peak.prominence):
        if peak.snr >= min_snr and all(
            abs(peak.center - other.center) >= min_distance for other in selected
        ):
            selected.append(peak)
    return selected


def _validate_inputs(
    detune: NDArray[np.float64],
    signal: NDArray[np.float64],
    prominence: float,
    distance: float,
) -> None:
    if (
        detune.ndim != 1
        or signal.shape != detune.shape
        or detune.size < 2
        or detune.dtype != np.float64
        or signal.dtype != np.float64
    ):
        raise ValueError("Expected matching float64 vectors of at least two points")
    if not (np.all(np.isfinite(detune)) and np.all(np.isfinite(signal))):
        raise ValueError("Axis and signal must be finite")
    steps = np.diff(detune)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError("Axis must be strictly monotonic")
    if not 0 <= prominence <= 1:
        raise ValueError("min_prominence must be in [0, 1]")
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("min_distance must be finite and nonnegative")
