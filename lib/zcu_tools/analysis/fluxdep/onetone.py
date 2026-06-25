"""One-tone peak detection rules for Flux-Dependence Analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from zcu_tools.utils.process import smooth_signal1d


def _require_spectrum(
    signals: NDArray[np.complex128], freqs: NDArray[np.float64]
) -> None:
    if signals.ndim != 2:
        raise ValueError("signals must be a 2D array")
    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1D axis")
    if freqs.size < 2:
        raise ValueError("freqs must contain at least two values")
    if signals.shape[1] != freqs.size:
        raise ValueError("signals second dimension must match len(freqs)")


def max_dispersion_freq_index(
    signals: NDArray[np.complex128], freqs: NDArray[np.float64]
) -> int:
    """Index of the frequency with the largest mean relative dispersion."""

    _require_spectrum(signals, freqs)
    abs_grad = (
        np.abs(signals[:, 1:] - signals[:, :-1]) / ((freqs[1:] - freqs[:-1])[None])
    )
    rel_grad = abs_grad / np.clip(np.abs(signals[:, 1:] + signals[:, :-1]), 1e-12, None)
    rel_grad = smooth_signal1d(rel_grad, method="wavelet", sigma=1.0, axis=1)
    return min(int(np.argmax(np.mean(rel_grad, axis=0))) + 1, len(freqs) - 1)


def smoothed_slice(
    signals: NDArray[np.complex128], freq_idx: int
) -> NDArray[np.float64]:
    """Normalised, inverted, smoothed amplitude slice at ``freq_idx``."""

    if signals.ndim != 2:
        raise ValueError("signals must be a 2D array")
    if not 0 <= freq_idx < signals.shape[1]:
        raise ValueError("freq_idx must be within the frequency axis")

    real_slice = np.abs(signals)[:, freq_idx]
    smoothed = smooth_signal1d(
        np.max(real_slice) - real_slice, method="wavelet", sigma=1.0
    )
    std = np.std(smoothed)
    if std == 0.0:
        # Flat slices carry no resonance dip; return finite zeros so peak
        # detection deterministically yields no points.
        return np.zeros_like(smoothed)
    return smoothed / std


def detect_peaks(smoothed: NDArray[np.float64], threshold: float) -> NDArray[np.intp]:
    """Peak indices of ``smoothed`` with prominence at least ``threshold``."""

    if smoothed.ndim != 1:
        raise ValueError("smoothed must be a 1D array")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    peaks, _ = find_peaks(smoothed, prominence=threshold)
    return peaks


def onetone_peak_points(
    signals: NDArray[np.complex128],
    dev_values: NDArray[np.float64],
    freqs: NDArray[np.float64],
    threshold: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    int,
    NDArray[np.float64],
    NDArray[np.intp],
]:
    """Detect one-tone peak points and return data useful to plotting adapters."""

    _require_spectrum(signals, freqs)
    if dev_values.ndim != 1:
        raise ValueError("dev_values must be a 1D axis")
    if dev_values.size != signals.shape[0]:
        raise ValueError("signals first dimension must match len(dev_values)")

    freq_idx = max_dispersion_freq_index(signals, freqs)
    smoothed = smoothed_slice(signals, freq_idx)
    peaks = detect_peaks(smoothed, threshold)
    s_dev_values = dev_values[peaks]
    s_freqs = np.full_like(s_dev_values, freqs[freq_idx])
    return s_dev_values, s_freqs, freq_idx, smoothed, peaks
