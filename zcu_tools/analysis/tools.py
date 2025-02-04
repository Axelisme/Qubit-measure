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
    # if all values are nan along the axis, replace with 0
    nan_mask = np.all(np.isnan(amps2D), axis=axis, keepdims=True)
    nan_mask = np.broadcast_to(nan_mask, amps2D.shape)
    amps2D[nan_mask] = 0

    if amps2D.dtype == np.complex:
        # if complex, minus the mean
        amps2D = np.abs(amps2D - np.nanmean(amps2D, axis=axis, keepdims=True))
    else:
        # if real, minus the median
        amps2D = amps2D - np.nanmedian(amps2D, axis=axis, keepdims=True)

    if rescale:
        # divide by the standard deviation
        stds = np.nanstd(amps2D, axis=axis, keepdims=True)
        stds[stds == 0] = 1e-10
        amps2D = amps2D / stds

    # restore nan values
    amps2D[nan_mask] = np.nan

    return amps2D


def rotate_phase(fpts, y, phase_slope):
    Is, Qs = y.real, y.imag

    angles = fpts * phase_slope * np.pi / 180
    Is_rot = Is * np.cos(angles) - Qs * np.sin(angles)
    Qs_rot = Is * np.sin(angles) + Qs * np.cos(angles)

    return Is_rot + 1j * Qs_rot
