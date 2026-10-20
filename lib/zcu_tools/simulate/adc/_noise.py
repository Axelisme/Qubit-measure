"""Receiver covariance factorization shared by means, traces and raw shots."""

import numpy as np

from .model import RealArray


def noise_root(covariance: RealArray) -> RealArray:
    """Factor a validated PSD float64 (2,2) I/Q covariance.

    Return an owned (2,2) root R with R @ R.T equal to covariance. Zero and
    singular covariances are valid; only eigensolver roundoff is clipped.
    Receiver or the statistics boundary owns finite/symmetry/PSD validation.
    """
    eigenvalues, vectors = np.linalg.eigh(covariance)
    return vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
