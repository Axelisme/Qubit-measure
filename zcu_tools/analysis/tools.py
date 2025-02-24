import numpy as np


def rotate2real(signals: np.ndarray):
    """
    Rotate the signals to maximize the contrast on real axis

    Parameters
    ----------
    signals : np.ndarray
        The 1-D complex signals

    Returns
    -------
    np.ndarray
        The rotated signals
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


def minus_mean(signals: np.ndarray, axis=None) -> np.ndarray:
    signals = signals.copy()  # prevent in-place modification

    if np.all(np.isnan(signals)):
        return signals

    if axis is None:
        signals -= np.nanmean(signals)

    else:
        _signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        # minus the mean
        where = ~np.all(np.isnan(_signals), axis=0)
        _signals[:, where] -= np.nanmean(_signals[:, where], axis=0, keepdims=True)

        signals = np.swapaxes(_signals, 0, axis)  # move the axis back

    return signals


def rescale(signals: np.ndarray, axis=None) -> np.ndarray:
    signals = signals.copy()  # prevent in-place modification
    nan_mask = np.isnan(signals)

    if np.all(nan_mask):
        return signals

    if axis is None:
        if np.sum(~nan_mask) > 1:  # at least 2 non-nan values
            signals /= np.nanstd(signals)

    else:
        _signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        where = np.sum(~np.isnan(_signals), axis=0) > 1
        _signals[:, where] /= np.nanstd(_signals[:, where], axis=0, keepdims=True)

        signals = np.swapaxes(_signals, 0, axis)  # move the axis back

    return signals


def rotate_phase(fpts, y, phase_slope):
    Is, Qs = y.real, y.imag

    angles = fpts * phase_slope * np.pi / 180
    Is_rot = Is * np.cos(angles) - Qs * np.sin(angles)
    Qs_rot = Is * np.sin(angles) + Qs * np.cos(angles)

    return Is_rot + 1j * Qs_rot
