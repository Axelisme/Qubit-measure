"""One-tone peak detection rules for Flux-Dependence Analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs
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


@dataclass(frozen=True, slots=True)
class OneToneInputs:
    """Captured, validated spectrum and reusable one-tone preprocessing.

    spectrum owns read-only signals/device values/GHz frequencies.
    max_freq_index identifies the maximum-dispersion frequency in spectrum.
    smoothed is its read-only, normalized inverted amplitude over the device axis.
    Construction raises ValueError for invalid spectrum dimensions or axes.
    """

    spectrum: FluxPickInputs
    max_freq_index: int = field(init=False)
    smoothed: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True, kw_only=True)
class OneTonePickState:
    """Complete committed peak selection against one captured OneToneInputs.

    threshold is finite prominence in [0, 5], dimensionless.
    peak_indices are unique ascending nonnegative indices into the device axis.
    Invalid threshold or index structure raises ValueError at construction.
    Input-dependent index bounds are checked by analyze_onetone_pick.
    """

    threshold: float
    peak_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class OneTonePickResult:
    """Uncalibrated selected points, including a valid empty selection.

    dev_values and freqs are equal-length 1D float64 arrays in captured axis order.
    dev_values uses native device units; freqs uses GHz at the captured slice.
    Flux calibration and sorting belong to the app's PointsService.
    """

    dev_values: NDArray[np.float64]
    freqs: NDArray[np.float64]


def pick_onetone_state(inputs: OneToneInputs, threshold: float) -> OneTonePickState:
    """Compute complete peak indices at finite dimensionless threshold [0, 5].

    Reuse captured preprocessing without writes; ValueError rejects invalid
    threshold before computation. No worker or presentation side effects.
    """
    raise NotImplementedError


def analyze_onetone_pick(
    inputs: OneToneInputs, state: OneTonePickState
) -> OneTonePickResult:
    """Project committed indices to detached device/GHz point arrays.

    ValueError rejects indices outside the captured device axis. Empty selection
    is valid; does not sort, calibrate flux, mutate inputs or decide publication.
    """
    raise NotImplementedError
