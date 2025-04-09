import warnings
from typing import Literal, Tuple

import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter1d


def rotate2real(signals: ndarray, ret_angle=False):
    """
    Rotate the signals to maximize the contrast on real axis by performing
    principal component analysis (PCA) on complex data

    Parameters
    ----------
    signals : ndarray
        The complex signals to be rotated. Must be a array of complex values.
    ret_angle : bool, default=False
        If True, return the rotation angle in radians.

    Returns
    -------
    Tuple[ndarray, float]
        The rotated signals with maximum variance aligned along the real axis
        and the rotation angle in radians.

    Notes
    -----
    This function:
    1. Calculates the covariance matrix between real and imaginary parts
    2. Finds the eigenvector corresponding to the largest eigenvalue
    3. Rotates the signal to align this principal component with the real axis
    """

    if signals.dtype != complex:
        raise ValueError(f"Expect complex signals, but get dtype {signals.dtype}")

    orig_shape = signals.shape
    if len(orig_shape) != 1:
        signals = signals.flatten()

    val_signals = signals[~np.isnan(signals)]

    if len(val_signals) < 2:
        return signals

    # calculate the covariance matrix
    cov = np.cov(val_signals.real, val_signals.imag)  # (2, 2)

    # calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)  # (2,), (2, 2)

    # sort the eigenvectors by decreasing eigenvalues
    eigenvectors = eigenvectors[:, eigenvalues.argmax()]  # (2,)

    if eigenvectors[0] < 0:
        eigenvectors = -eigenvectors  # make rotation angle from -90 to 90 deg

    # rotate the signals to maximize the contrast on real axis
    rot_signals = signals * eigenvectors.dot([1, -1j])

    if len(orig_shape) != 1:
        rot_signals = rot_signals.reshape(orig_shape)

    if ret_angle:
        return rot_signals, np.arctan2(eigenvectors[1], eigenvectors[0])
    return rot_signals


def minus_background(signals: ndarray, axis=None, method="median") -> ndarray:
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


def minus_median(signals: ndarray, axis=None) -> ndarray:
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
            signals.real -= np.nanmedian(signals.real)
            signals.imag -= np.nanmedian(signals.imag)
        else:
            signals -= np.nanmedian(signals)

    elif isinstance(axis, int):
        signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        # minus the median
        val_mask = ~np.all(np.isnan(signals), axis=0)
        val_signals = signals[:, val_mask]
        if val_signals.dtype == complex:
            val_signals.real -= np.nanmedian(val_signals.real, axis=0, keepdims=True)
            val_signals.imag -= np.nanmedian(val_signals.imag, axis=0, keepdims=True)
        else:
            val_signals -= np.nanmedian(val_signals, axis=0, keepdims=True)
        signals[:, val_mask] = val_signals

        signals = np.swapaxes(signals, 0, axis)  # move the axis back

    else:
        raise ValueError(f"Invalid axis: {axis} for minus_median")

    return signals


def minus_mean(signals: ndarray, axis=None) -> ndarray:
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
        signals -= np.nanmean(signals)

    elif isinstance(axis, int):
        signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        # minus the mean
        val_mask = ~np.all(np.isnan(signals), axis=0)
        signals[:, val_mask] -= np.nanmean(signals[:, val_mask], axis=0, keepdims=True)

        signals = np.swapaxes(signals, 0, axis)  # move the axis back

    else:
        raise ValueError(f"Invalid axis: {axis} for minus_mean")

    return signals


def rescale(signals: ndarray, axis=None) -> ndarray:
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
            signals /= np.nanstd(signals)

    elif isinstance(axis, int):
        signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        val_mask = np.sum(~np.isnan(signals), axis=0) > 1
        signals[:, val_mask] /= np.nanstd(signals[:, val_mask], axis=0, keepdims=True)

        signals = np.swapaxes(signals, 0, axis)  # move the axis back
    else:
        raise ValueError(f"Invalid axis: {axis} for rescale")

    return signals


def calculate_noise(signals: ndarray) -> Tuple[float, ndarray]:
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

    return np.abs(signals - m_signals).mean(), m_signals


def peak_n_avg(data: ndarray, n: int, mode: Literal["max", "min"] = "max"):
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
        return np.nanmean(data)

    peak_fn = np.nanargmax if mode == "max" else np.nanargmin

    peaks = np.empty(n, dtype=data.dtype)
    _data = data.copy().flatten()  # prevent in-place modification
    for i in range(n):
        peak_idx = peak_fn(_data)
        peaks[i], _data[peak_idx] = _data[peak_idx], np.nan

    return np.mean(peaks)


def rotate_phase(fpts, signals, phase_slope):
    """
    Rotate the phase of complex signals based on frequency points

    Parameters
    ----------
    fpts : ndarray
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
    The rotation angle is calculated as: angle = fpts * phase_slope * π/180
    """
    Is, Qs = signals.real, signals.imag

    angles = fpts * phase_slope * np.pi / 180
    Is_rot = Is * np.cos(angles) - Qs * np.sin(angles)
    Qs_rot = Is * np.sin(angles) + Qs * np.cos(angles)

    return Is_rot + 1j * Qs_rot
