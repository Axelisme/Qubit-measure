from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d

T_dtype = TypeVar("T_dtype", bound=np.generic)
SmoothMethod: TypeAlias = Literal["wavelet", "gaussian"]
WaveletThresholdMode: TypeAlias = Literal["soft", "hard"]


def find_rotate_angle(signals: NDArray[np.complex128]) -> float:
    if signals.dtype != np.complex128:
        raise ValueError(f"Expect complex signals, but get dtype {signals.dtype}")

    orig_shape = signals.shape
    if len(orig_shape) != 1:
        signals = signals.flatten()

    val_signals = signals[~np.isnan(signals)]

    if len(val_signals) < 2:
        raise ValueError(
            f"At least 2 non-NaN values are required, but get {len(val_signals)}"
        )

    # calculate the covariance matrix
    cov = np.cov(val_signals.real, val_signals.imag)  # (2, 2)

    # calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)  # (2,), (2, 2)

    # sort the eigenvectors by decreasing eigenvalues
    eigenvectors = eigenvectors[:, eigenvalues.argmax()]  # (2,)

    if eigenvectors[0] < 0:
        eigenvectors = -eigenvectors  # make rotation angle from -90 to 90 deg

    return np.arctan2(eigenvectors[1], eigenvectors[0])


def rotate2real(signals: NDArray[np.complex128]) -> NDArray[np.complex128]:
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return signals

    return signals * np.exp(-1j * angle)


def minus_background(
    signals: NDArray[T_dtype], axis=None, method="median"
) -> NDArray[T_dtype]:
    """
    Subtract the background from the signals

    Parameters
    ----------
    signals : ndarray
        The signals to process, can be 1-D or 2-D
    axis : int, None
        The axis to process, if None, process the whole signals
    method : str
        The method to calculate the background, 'median' or 'mean'

    Returns
    -------
    ndarray
        The signals with background subtracted
    """

    if method == "median":
        return minus_median(signals, axis)
    elif method == "mean":
        return minus_mean(signals, axis)
    else:
        raise ValueError(f"Invalid method: {method}")


def minus_median(signals: NDArray[T_dtype], axis=None) -> NDArray[T_dtype]:
    """
    Subtract the median from signals, useful for background removal

    Parameters
    ----------
    signals : ndarray
        The signals array to process. Can be real or complex valued.
    axis : int, optional
        The axis along which to calculate the median.
        If None, calculate the median of the entire array.

    Returns
    -------
    ndarray
        A copy of the signals with the median subtracted. Original
        array is not modified.

    Notes
    -----
    For complex arrays, real and imaginary parts are processed separately.
    NaN values are handled using numpy's nanmedian function.
    """
    signals = signals.copy()  # prevent in-place modification

    if np.all(np.isnan(signals)):
        return signals

    if axis is None:
        if signals.dtype == complex:  # perform on real & imag part
            signals.real -= np.nanmedian(signals.real)  # type: ignore
            signals.imag -= np.nanmedian(signals.imag)  # type: ignore
        else:
            signals -= np.nanmedian(signals)  # type: ignore

    elif isinstance(axis, int):
        signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        # minus the median
        val_mask = ~np.all(np.isnan(signals), axis=0)
        val_signals = signals[:, val_mask]
        if val_signals.dtype == complex:
            val_signals.real -= np.nanmedian(val_signals.real, axis=0, keepdims=True)  # type: ignore
            val_signals.imag -= np.nanmedian(val_signals.imag, axis=0, keepdims=True)  # type: ignore
        else:
            val_signals -= np.nanmedian(val_signals, axis=0, keepdims=True)  # type: ignore
        signals[:, val_mask] = val_signals

        signals = np.swapaxes(signals, 0, axis)  # move the axis back

    else:
        raise ValueError(f"Invalid axis: {axis} for minus_median")

    return signals


def minus_mean(signals: NDArray[T_dtype], axis=None) -> NDArray[T_dtype]:
    """
    Subtract the mean from signals, useful for baseline correction

    Parameters
    ----------
    signals : ndarray
        The signals array to process. Can be real or complex valued.
    axis : int, optional
        The axis along which to calculate the mean.
        If None, calculate the mean of the entire array.

    Returns
    -------
    ndarray
        A copy of the signals with the mean subtracted. Original
        array is not modified.

    Notes
    -----
    NaN values are handled using numpy's nanmean function.
    """
    signals = signals.copy()  # prevent in-place modification

    if np.all(np.isnan(signals)):
        return signals

    if axis is None:
        signals -= np.nanmean(signals)  # type: ignore

    elif isinstance(axis, int):
        signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        # minus the mean
        val_mask = ~np.all(np.isnan(signals), axis=0)
        signals[:, val_mask] -= np.nanmean(signals[:, val_mask], axis=0, keepdims=True)  # type: ignore

        signals = np.swapaxes(signals, 0, axis)  # move the axis back

    else:
        raise ValueError(f"Invalid axis: {axis} for minus_mean")

    return signals


def rescale(signals: NDArray[T_dtype], axis: int | None = None) -> NDArray[T_dtype]:
    """
    Rescale signals by dividing by the standard deviation

    Parameters
    ----------
    signals : ndarray
        The signals array to process. Must be real valued (not complex).
    axis : int, optional
        The axis along which to calculate the standard deviation.
        If None, calculate the standard deviation of the entire array.

    Returns
    -------
    ndarray
        A copy of the signals rescaled by standard deviation. Original
        array is not modified.

    Notes
    -----
    - This function does not support complex signals and will return the
      original array with a warning if complex input is provided.
    - NaN values are handled using numpy's nanstd function.
    - At least 2 non-NaN values are required for rescaling.
    """
    signals = signals.copy()  # prevent in-place modification

    if signals.dtype == complex:
        warnings.warn("Rescale complex signals is not supported, do nothing")
        return signals

    if np.all(np.isnan(signals)):
        return signals

    if axis is None:
        if np.sum(~np.isnan(signals)) > 1:  # at least 2 non-nan values
            signals /= np.nanstd(signals)  # type: ignore

    elif isinstance(axis, int):
        signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        val_mask = np.sum(~np.isnan(signals), axis=0) > 1
        signals[:, val_mask] /= np.nanstd(signals[:, val_mask], axis=0, keepdims=True)  # type: ignore

        signals = np.swapaxes(signals, 0, axis)  # move the axis back
    else:
        raise ValueError(f"Invalid axis: {axis} for rescale")

    return signals


def _require_pywt() -> Any:
    try:
        import pywt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Wavelet smoothing requires PyWavelets; install the client extra."
        ) from exc
    return pywt


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError("axis is invalid for a scalar signal")
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for signal ndim {ndim}")
    return normalized


def _normalize_axes(axes: Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(ndim))
    normalized = tuple(_normalize_axis(axis, ndim) for axis in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"axes must not contain duplicates: {axes}")
    return normalized


def _resolve_wavelet_level(pywt: Any, length: int, wavelet: str, level: int) -> int:
    if length < 2:
        return 0
    max_level = int(pywt.dwt_max_level(length, pywt.Wavelet(wavelet).dec_len))
    if max_level <= 0:
        return 0
    if level <= 0:
        return min(3, max_level)
    if level > max_level:
        raise ValueError(
            f"wavelet_level={level} is too high for length {length}; max is {max_level}"
        )
    return level


def _wavelet_denoise_real_vector(
    values: NDArray[np.float64],
    *,
    pywt: Any,
    wavelet: str,
    level: int,
    threshold_scale: float,
    threshold_mode: WaveletThresholdMode,
) -> NDArray[np.float64]:
    if threshold_scale < 0:
        raise ValueError("threshold_scale must be non-negative")

    out = values.astype(np.float64, copy=True)
    finite_mask = np.isfinite(out)
    if np.count_nonzero(finite_mask) < 2:
        return out

    invalid_mask = ~finite_mask
    if np.any(invalid_mask):
        x = np.arange(out.size)
        out[invalid_mask] = np.interp(x[invalid_mask], x[finite_mask], out[finite_mask])

    resolved_level = _resolve_wavelet_level(pywt, out.size, wavelet, level)
    if resolved_level == 0:
        out[invalid_mask] = values[invalid_mask]
        return out

    coeffs = pywt.wavedec(out, wavelet=wavelet, mode="symmetric", level=resolved_level)
    detail = np.asarray(coeffs[-1], dtype=np.float64)
    noise_sigma = float(np.median(np.abs(detail - np.median(detail))) / 0.6745)
    if np.isfinite(noise_sigma) and noise_sigma > 0.0 and threshold_scale > 0.0:
        threshold = threshold_scale * noise_sigma * np.sqrt(2.0 * np.log(out.size))
        coeffs[1:] = [
            pywt.threshold(coeff, threshold, mode=threshold_mode)
            for coeff in coeffs[1:]
        ]

    reconstructed = np.asarray(
        pywt.waverec(coeffs, wavelet=wavelet, mode="symmetric"),
        dtype=np.float64,
    )
    if reconstructed.size < out.size:
        reconstructed = np.pad(reconstructed, (0, out.size - reconstructed.size))
    out = reconstructed[: out.size]
    out[invalid_mask] = values[invalid_mask]
    return out


def wavelet_denoise1d(
    signals: NDArray[Any],
    *,
    axis: int = -1,
    wavelet: str = "sym4",
    level: int = 0,
    threshold_scale: float = 1.0,
    threshold_mode: WaveletThresholdMode = "soft",
) -> NDArray[Any]:
    """Denoise 1D traces along one axis with wavelet coefficient thresholding."""
    pywt = _require_pywt()
    signals_arr = np.asarray(signals)
    if signals_arr.ndim == 0:
        return signals_arr.copy()
    normalized_axis = _normalize_axis(axis, signals_arr.ndim)

    if np.iscomplexobj(signals_arr):
        real = wavelet_denoise1d(
            np.real(signals_arr),
            axis=normalized_axis,
            wavelet=wavelet,
            level=level,
            threshold_scale=threshold_scale,
            threshold_mode=threshold_mode,
        )
        imag = wavelet_denoise1d(
            np.imag(signals_arr),
            axis=normalized_axis,
            wavelet=wavelet,
            level=level,
            threshold_scale=threshold_scale,
            threshold_mode=threshold_mode,
        )
        return real + 1j * imag

    moved = np.moveaxis(signals_arr.astype(np.float64, copy=True), normalized_axis, -1)
    out = np.empty_like(moved)
    for idx in np.ndindex(moved.shape[:-1]):
        out[idx] = _wavelet_denoise_real_vector(
            moved[idx],
            pywt=pywt,
            wavelet=wavelet,
            level=level,
            threshold_scale=threshold_scale,
            threshold_mode=threshold_mode,
        )
    return np.moveaxis(out, -1, normalized_axis)


def smooth_signal1d(
    signals: NDArray[Any],
    *,
    method: SmoothMethod = "wavelet",
    sigma: float = 1.0,
    axis: int = -1,
    wavelet: str = "sym4",
    wavelet_level: int = 0,
    wavelet_threshold: float | None = None,
    threshold_mode: WaveletThresholdMode = "soft",
) -> NDArray[Any]:
    """Smooth traces along one axis with a shared method knob."""
    if method == "gaussian":
        return cast(NDArray[Any], gaussian_filter1d(signals, sigma=sigma, axis=axis))
    elif method == "wavelet":
        return wavelet_denoise1d(
            signals,
            axis=axis,
            wavelet=wavelet,
            level=wavelet_level,
            threshold_scale=sigma if wavelet_threshold is None else wavelet_threshold,
            threshold_mode=threshold_mode,
        )
    else:
        raise ValueError(f"Invalid smoothing method: {method}")


def smooth_signal_nd(
    signals: NDArray[Any],
    *,
    method: SmoothMethod = "wavelet",
    sigma: float = 1.0,
    axes: Sequence[int] | None = None,
    wavelet: str = "sym4",
    wavelet_level: int = 0,
    wavelet_threshold: float | None = None,
    threshold_mode: WaveletThresholdMode = "soft",
) -> NDArray[Any]:
    """Smooth an N-D signal; wavelet mode applies separable 1D denoise per axis."""
    if method == "gaussian":
        if axes is None:
            return cast(NDArray[Any], gaussian_filter(signals, sigma=sigma))
        return cast(
            NDArray[Any], gaussian_filter(signals, sigma=sigma, axes=tuple(axes))
        )
    elif method == "wavelet":
        out = np.asarray(signals)
        for axis in _normalize_axes(axes, out.ndim):
            out = wavelet_denoise1d(
                out,
                axis=axis,
                wavelet=wavelet,
                level=wavelet_level,
                threshold_scale=sigma
                if wavelet_threshold is None
                else wavelet_threshold,
                threshold_mode=threshold_mode,
            )
        return out
    else:
        raise ValueError(f"Invalid smoothing method: {method}")


def calculate_noise(signals: NDArray[T_dtype]) -> tuple[float, NDArray[T_dtype]]:
    """
    Calculate the noise level of the signals
    by comparing the signals with a smoothed version of themselves
    using Gaussian filtering.

    Parameters
    ----------
    signals : ndarray
        The signals array to process. Can be real or complex valued.

    Returns
    -------
    float
        The noise level of the signals, defined as the mean absolute difference
        between the original signals and the smoothed signals.
    ndarray
        The smoothed signals obtained by applying Gaussian filtering.
    """
    m_signals = gaussian_filter1d(signals, sigma=1)
    m_signals = cast(NDArray[T_dtype], m_signals)

    return np.abs(np.subtract(signals, m_signals)).mean(), m_signals


def peak_n_avg(
    data: NDArray[np.float64], n: int, mode: Literal["max", "min"] = "max"
) -> float:
    """
    Find the first n max/min points in the data, and return their average
    Parameters
    ----------
    data : ndarray
        The data to process.
    n : int
        The number of points to find.
    mode : str
        The mode to find, either "max" or "min". Default is "max".
    Returns

    -------
    float
        The average of the first n max/min points.
    """

    assert mode in ["max", "min"], f"Invalid mode: {mode}"

    if n <= 0:
        raise ValueError(f"n should be positive, but get {n}")

    if np.sum(~np.isnan(data)) <= n:
        return np.nanmean(data)  # type: ignore

    # Replace NaN with a sentinel that loses the argpartition race so NaN
    # positions are never selected as top-n candidates.
    # For "max": NaN -> -inf (guaranteed smaller than any finite value).
    # For "min": NaN -> +inf (guaranteed larger than any finite value).
    sentinel = -np.inf if mode == "max" else np.inf
    flat = data.flatten()
    masked = np.where(np.isnan(flat), sentinel, flat)

    if mode == "max":
        # argpartition(-n:) contains indices of the n largest values in O(N).
        # The slice is unordered, but we only need the mean — order doesn't matter.
        top_indices = np.argpartition(masked, -n)[-n:]
    else:
        top_indices = np.argpartition(masked, n)[:n]

    return float(np.mean(flat[top_indices]))


def rotate_phase(
    freqs: NDArray[np.float64], signals: NDArray[np.complex128], phase_slope: float
) -> NDArray[np.complex128]:
    """
    Rotate the phase of complex signals based on frequency points

    Parameters
    ----------
    freqs : ndarray
        Frequency points array
    signals : ndarray
        Complex signal array to be phase-rotated
    phase_slope : float
        Phase rotation slope in degrees per frequency unit

    Returns
    -------
    ndarray
        Phase-rotated complex signals

    Notes
    -----
    This function applies a frequency-dependent phase rotation to complex signals.
    The rotation angle is calculated as: angle = freqs * phase_slope * π/180
    """
    Is, Qs = signals.real, signals.imag

    angles = freqs * phase_slope * np.pi / 180
    Is_rot = Is * np.cos(angles) - Qs * np.sin(angles)
    Qs_rot = Is * np.sin(angles) + Qs * np.cos(angles)

    return Is_rot + 1j * Qs_rot
