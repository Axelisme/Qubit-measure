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


def NormalizeData(amps2D: np.ndarray, axis=None, rescale=True) -> np.ndarray:
    # find the mask of all values are nan along the axis,
    amps2D = amps2D.copy()  # prevent in-place modification
    nan_mask = np.isnan(amps2D)

    # skip if all values are nan
    if np.all(nan_mask):
        return amps2D

    _amps2D = np.swapaxes(amps2D, axis, 0)  # move the axis to the first dimension

    # minus the mean
    where = np.all(np.isnan(_amps2D), axis=0)
    _amps2D[:, ~where] -= np.nanmean(_amps2D[:, ~where], axis=0, keepdims=True)
    _amps2D = np.abs(_amps2D).astype(np.float64)

    if rescale:
        where = np.sum(_amps2D, axis=0) > 1
        _amps2D[:, where] /= np.std(_amps2D[:, where], axis=0)
        _amps2D[:, ~where] = 0

    amps2D = np.swapaxes(_amps2D, 0, axis)  # move the axis back

    # restore nan values
    amps2D[nan_mask] = np.nan

    return amps2D


def rotate_phase(fpts, y, phase_slope):
    Is, Qs = y.real, y.imag

    angles = fpts * phase_slope * np.pi / 180
    Is_rot = Is * np.cos(angles) - Qs * np.sin(angles)
    Qs_rot = Is * np.sin(angles) + Qs * np.cos(angles)

    return Is_rot + 1j * Qs_rot
