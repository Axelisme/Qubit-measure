"""Stateless processing functions for Flux-Dependence Analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal1d


def _require_1d_axis(axis: NDArray[np.float64], name: str) -> None:
    if axis.ndim != 1:
        raise ValueError(f"{name} must be a 1D axis")
    if axis.size < 2:
        raise ValueError(f"{name} must contain at least two values")
    if axis[-1] == axis[0]:
        raise ValueError(f"{name} endpoints must be distinct")


def cast2real_and_norm(
    signals: NDArray[np.complexfloating[Any, Any]],
    *,
    use_phase: bool = True,
    sigma: float = 1,
    smooth_method: SmoothMethod = "wavelet",
) -> NDArray[np.float64]:
    """Convert 2D complex spectra to real features without modifying the input.

    ``signals`` has one spectrum per row and frequency samples along columns.
    With ``use_phase=True``, subtract each row's complex mean, smooth along
    columns, take the magnitude, and divide by that magnitude's row standard
    deviation. Constant rows use divisor 1. With ``use_phase=False``, smooth
    the input magnitudes without centering or normalization.

    ``smooth_method`` selects "wavelet" coefficient thresholding or "gaussian"
    filtering. ``sigma`` is the wavelet threshold scale or the positive Gaussian
    width in samples. All options must be passed by keyword. Return a real-valued
    array with the same shape as signals.

    Raise ValueError for non-2D input or an unsupported smoothing method.
    Smoothing errors propagate; wavelet smoothing requires PyWavelets and raises
    RuntimeError when it is unavailable.
    """

    if signals.ndim != 2:
        raise ValueError("signals must be a 2D array")

    if use_phase:
        centered = signals - np.ma.mean(signals, axis=1, keepdims=True)
        smoothed = smooth_signal1d(
            centered,
            options=SmoothingOptions(method=smooth_method, sigma=sigma),
            axis=1,
        )
        real_signals = np.abs(smoothed)
        std = np.ma.std(real_signals, axis=1, keepdims=True)
        # Constant rows carry no spectral feature. Keeping their divisor at 1.0
        # avoids inf/nan values that would later poison peak detection.
        std = np.where(np.ma.getdata(std) > 0, std, 1.0)
        real_signals /= std
    else:
        real_signals = np.abs(signals)
        real_signals = smooth_signal1d(
            real_signals,
            options=SmoothingOptions(method=smooth_method, sigma=sigma),
            axis=1,
        )

    return real_signals


@dataclass(frozen=True)
class OneToneBackground:
    """A OneTone spectrum's display image and its colour limits.

    values is the dip depth shaped like the spectrum (device, frequency): how
    far each pixel's smoothed magnitude lies below the largest one, so a
    resonator dip holds the largest values. NaN pixels stay NaN. vmin and
    vmax are the ``contrast_limits`` of values, with vmin < vmax.
    """

    values: NDArray[np.float64]
    vmin: float
    vmax: float


def onetone_background(signals: NDArray[np.complex128]) -> OneToneBackground:
    """Return the dip-depth image and colour limits every OneTone view draws.

    signals is the complex spectrum shaped (device, frequency). The magnitude
    is smoothed along frequency (``cast2real_and_norm`` without phase) and
    subtracted from its largest finite value. Under ``gray_r`` with the
    returned limits the dip draws dark on a light background. ValueError
    rejects signals that are not 2D.
    """
    magnitude = cast2real_and_norm(signals, use_phase=False)
    depth = np.nanmax(magnitude) - magnitude
    vmin, vmax = contrast_limits(depth)
    return OneToneBackground(values=depth, vmin=vmin, vmax=vmax)


def contrast_limits(values: NDArray[np.float64]) -> tuple[float, float]:
    """Robust display limits (2nd–98th percentile) to boost feature contrast.

    Clipping the colour range to percentiles (instead of min/max) keeps a few
    outliers from compressing the dynamic range, so the spectral feature stands
    out against the background. NaNs are ignored. The returned lower limit is
    always below the upper one:

    - no finite value gives ``(0, 1)``;
    - coinciding percentiles widen to the finite ``(min, max)``;
    - a constant finite value ``c`` gives ``(c, c + 1)``.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    if hi > lo:
        return float(lo), float(hi)
    lowest, highest = float(finite.min()), float(finite.max())
    if highest > lowest:
        return lowest, highest
    return lowest, lowest + 1.0


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
    xs: NDArray[np.float64], data: NDArray[np.float64], center: float
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
