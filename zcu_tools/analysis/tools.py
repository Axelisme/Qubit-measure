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


def NormalizeData(signals2D: np.ndarray, axis=None, rescale=True) -> np.ndarray:
    signals2D = signals2D - np.nanmedian(signals2D, axis=axis, keepdims=True)
    if rescale:
        mins = np.nanmin(signals2D, axis=axis, keepdims=True)
        maxs = np.nanmax(signals2D, axis=axis, keepdims=True)
        signals2D /= maxs - mins
    return signals2D


def rotate_phase(fpts, y, phase_slope):
    Is, Qs = y.real, y.imag

    angles = fpts * phase_slope * np.pi / 180
    Is_rot = Is * np.cos(angles) - Qs * np.sin(angles)
    Qs_rot = Is * np.sin(angles) + Qs * np.cos(angles)

    return Is_rot + 1j * Qs_rot
