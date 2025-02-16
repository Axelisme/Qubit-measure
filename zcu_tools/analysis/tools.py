import numpy as np


def convert2max_contrast(Is: np.ndarray, Qs: np.ndarray):
    """
    rotate the 2-d input data to maximize on the x-axis
    """

    # calculate the covariance matrix
    cov = np.cov(Is, Qs)

    # calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # sort the eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # rotate the data
    data = np.vstack([Is, Qs])
    data_rot = np.dot(eigenvectors.T, data)

    return data_rot[0], data_rot[1]


def NormalizeData(signals: np.ndarray, axis=None, rescale=True) -> np.ndarray:
    # find the mask of all values are nan along the axis,
    signals = signals.copy()  # prevent in-place modification
    nan_mask = np.isnan(signals)

    # skip if all values are nan
    if np.all(nan_mask):
        return signals

    if axis is None:
        signals -= np.nanmean(signals)
        amps = np.abs(signals).astype(np.float64)

        if rescale:
            amps /= np.nanstd(amps)

    else:
        _signals = np.swapaxes(signals, axis, 0)  # move the axis to the first dimension

        # minus the mean
        where = np.all(np.isnan(_signals), axis=0)
        _signals[:, ~where] -= np.nanmean(_signals[:, ~where], axis=0, keepdims=True)
        _amps = np.abs(_signals).astype(np.float64)

        if rescale:
            where = np.sum(~np.isnan(_amps), axis=0) > 1
            _amps[:, where] /= np.nanstd(_amps[:, where], axis=0, keepdims=True)

        amps = np.swapaxes(_amps, 0, axis)  # move the axis back

    return amps


def rotate_phase(fpts, y, phase_slope):
    Is, Qs = y.real, y.imag

    angles = fpts * phase_slope * np.pi / 180
    Is_rot = Is * np.cos(angles) - Qs * np.sin(angles)
    Qs_rot = Is * np.sin(angles) + Qs * np.cos(angles)

    return Is_rot + 1j * Qs_rot
