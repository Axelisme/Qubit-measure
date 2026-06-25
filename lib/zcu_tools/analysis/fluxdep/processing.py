"""Stateless processing functions for Flux-Dependence Analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from zcu_tools.utils.process import SmoothMethod, smooth_signal1d


def _require_1d_axis(axis: NDArray[np.float64], name: str) -> None:
    if axis.ndim != 1:
        raise ValueError(f"{name} must be a 1D axis")
    if axis.size < 2:
        raise ValueError(f"{name} must contain at least two values")
    if axis[-1] == axis[0]:
        raise ValueError(f"{name} endpoints must be distinct")


def cast2real_and_norm(
    signals: NDArray,
    use_phase: bool = True,
    sigma: float = 1,
    smooth_method: SmoothMethod = "wavelet",
) -> NDArray[np.float64]:
    """Convert complex spectra to real-valued, row-normalised features."""

    if signals.ndim != 2:
        raise ValueError("signals must be a 2D array")

    if use_phase:
        centered = signals - np.ma.mean(signals, axis=1, keepdims=True)
        smoothed = smooth_signal1d(centered, method=smooth_method, sigma=sigma, axis=1)
        real_signals = np.abs(smoothed)
        std = np.ma.std(real_signals, axis=1, keepdims=True)
        # Constant rows carry no spectral feature. Keeping their divisor at 1.0
        # avoids inf/nan values that would later poison peak detection.
        std = np.where(np.ma.getdata(std) > 0, std, 1.0)
        real_signals /= std
    else:
        real_signals = np.abs(signals)
        real_signals = smooth_signal1d(
            real_signals, method=smooth_method, sigma=sigma, axis=1
        )

    return real_signals


def spectrum2d_findpoint(
    dev_values: NDArray[np.float64],
    freqs: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    threshold: float,
    weight: NDArray[np.float64] | NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find up to three high peaks per device-value row in a 2D spectrum."""

    _require_1d_axis(dev_values, "dev_values")
    _require_1d_axis(freqs, "freqs")
    if real_signals.shape != (dev_values.size, freqs.size):
        raise ValueError("real_signals shape must match (len(dev_values), len(freqs))")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if weight is not None:
        if weight.shape != real_signals.shape:
            raise ValueError("weight shape must match real_signals")
        real_signals = real_signals * weight

    s_dev_values: list[float] = []
    s_freqs: list[float] = []
    for i in range(real_signals.shape[0]):
        peaks, _ = find_peaks(
            real_signals[i, :], height=threshold, width=(1, None), distance=5
        )

        if len(peaks) > 3:
            peaks = peaks[np.argsort(real_signals[i, peaks])[-3:]]

        s_dev_values.extend(float(dev_values[i]) for _ in peaks)
        s_freqs.extend(float(freqs[p]) for p in peaks)
    return np.array(s_dev_values, dtype=np.float64), np.array(s_freqs, dtype=np.float64)


def downsample_points(
    xs: NDArray[np.float64], ys: NDArray[np.float64], threshold: float
) -> NDArray[np.bool_]:
    """Select a deterministic sparse subset while preserving same-x groups."""

    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")
    if xs.ndim != 1:
        raise ValueError("xs and ys must be 1D arrays")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    selected: list[int] = []
    mask = np.zeros_like(xs, dtype=bool)

    idxs = np.random.Generator(np.random.PCG64(0)).permutation(len(xs))
    for i in idxs:
        if int(i) in selected:
            continue

        too_close = False
        for j in selected:
            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < threshold**2:
                too_close = True
                break

        if not too_close:
            for j in idxs:
                j_int = int(j)
                if xs[i] == xs[j] and j_int not in selected:
                    selected.append(j_int)
                    mask[j] = True

    return mask


def diff_mirror(
    xs: NDArray[np.float64], data: NDArray, center: float
) -> NDArray[np.float64]:
    """Absolute difference between data and its mirror around ``center``."""

    _require_1d_axis(xs, "xs")
    if data.shape[0] != xs.size:
        raise ValueError("data first dimension must match len(xs)")

    c_idx = (len(xs) - 1) * (center - xs[0]) / (xs[-1] - xs[0])
    idxs = np.arange(data.shape[0])
    mirror_idxs = np.round(2 * c_idx - idxs).astype(int)
    valid = (mirror_idxs >= 0) & (mirror_idxs < data.shape[0])
    diff_data = np.zeros_like(data, dtype=np.float64)

    # NaN can represent unmeasured/masked points. Let it propagate without
    # warning; downstream loss code decides how to ignore invalid rows.
    with np.errstate(invalid="ignore"):
        diff_data[valid] = np.abs(data[valid] - data[mirror_idxs[valid]])

    return diff_data
