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

    if np.iscomplexobj(amps2D):
        # if complex, minus the mean
        amps2D = np.abs(
            amps2D - np.nanmean(amps2D, axis=axis, keepdims=True, where=~nan_mask)
        )
    else:
        # if real, minus the median
        nan_row = np.all(nan_mask, axis=axis)
        amps2D[~nan_row] -= np.nanmedian(amps2D[~nan_row], axis=axis, keepdims=True)

    if rescale:
        # divide by the standard deviation
        stds = np.nanstd(amps2D, axis=axis, keepdims=True, where=~nan_mask)
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
