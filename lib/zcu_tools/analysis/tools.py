import warnings

import numpy as np


def rotate2real(signals: np.ndarray):
    """
    Rotate the signals to maximize the contrast on real axis by performing
    principal component analysis (PCA) on complex data

    Parameters
    ----------
    signals : np.ndarray
        The 1-D complex signals to be rotated. Must be a 1D array of complex values.

    Returns
    -------
    np.ndarray
        The rotated signals with maximum variance aligned along the real axis

    Notes
    -----
    This function:
    1. Calculates the covariance matrix between real and imaginary parts
    2. Finds the eigenvector corresponding to the largest eigenvalue
    3. Rotates the signal to align this principal component with the real axis
    """

    if len(signals.shape) != 1:
        raise ValueError(f"Expect 1-D signals, but get shape {signals.shape}")
    if signals.dtype != complex:
        raise ValueError(f"Expect complex signals, but get dtype {signals.dtype}")

    # calculate the covariance matrix
    cov = np.cov(signals.real, signals.imag)  # (2, 2)

    # calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)  # (2,), (2, 2)

    # sort the eigenvectors by decreasing eigenvalues
    eigenvectors = eigenvectors[:, eigenvalues.argmax()]  # (2,)

    # rotate the signals to maximize the contrast on real axis
    rot_signals = signals * eigenvectors.dot([1, -1j])

    return rot_signals


def minus_background(signals: np.ndarray, axis=None, method="median") -> np.ndarray:
    """
    Subtract the background from the signals

    Parameters
    ----------
    signals : np.ndarray
        The signals to process, can be 1-D or 2-D
    axis : int, None
        The axis to process, if None, process the whole signals
    method : str
        The method to calculate the background, 'median' or 'mean'

    Returns
    -------
    np.ndarray
        The signals with background subtracted
    """

    if method == "median":
        return minus_median(signals, axis)
    elif method == "mean":
        return minus_mean(signals, axis)
    else:
        raise ValueError(f"Invalid method: {method}")


def minus_median(signals: np.ndarray, axis=None) -> np.ndarray:
    """
    Subtract the median from signals, useful for background removal

    Parameters
    ----------
    signals : np.ndarray
        The signals array to process. Can be real or complex valued.
    axis : int, optional
        The axis along which to calculate the median.
        If None, calculate the median of the entire array.

    Returns
    -------
    np.ndarray
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


def minus_mean(signals: np.ndarray, axis=None) -> np.ndarray:
    """
    Subtract the mean from signals, useful for baseline correction

    Parameters
    ----------
    signals : np.ndarray
        The signals array to process. Can be real or complex valued.
    axis : int, optional
        The axis along which to calculate the mean.
        If None, calculate the mean of the entire array.

    Returns
    -------
    np.ndarray
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


def rescale(signals: np.ndarray, axis=None) -> np.ndarray:
    """
    Rescale signals by dividing by the standard deviation

    Parameters
    ----------
    signals : np.ndarray
        The signals array to process. Must be real valued (not complex).
    axis : int, optional
        The axis along which to calculate the standard deviation.
        If None, calculate the standard deviation of the entire array.

    Returns
    -------
    np.ndarray
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


def rotate_phase(fpts, signals, phase_slope):
    """
    Rotate the phase of complex signals based on frequency points

    Parameters
    ----------
    fpts : np.ndarray
        Frequency points array
    signals : np.ndarray
        Complex signal array to be phase-rotated
    phase_slope : float
        Phase rotation slope in degrees per frequency unit

    Returns
    -------
    np.ndarray
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
