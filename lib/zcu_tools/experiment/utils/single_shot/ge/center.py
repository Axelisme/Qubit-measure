import numpy as np

from .base import fitting_ge_and_plot


def get_rotate_angle(
    Ig: np.ndarray, Qg: np.ndarray, Ie: np.ndarray, Qe: np.ndarray
) -> dict:
    """
    Calculate the optimal rotation angle using the center method.

    This method determines the rotation angle by calculating the angle between
    the line connecting the medians of ground and excited state clusters and the I-axis.

    Parameters
    ----------
    Ig : np.ndarray
        I (in-phase) data for ground state.
    Qg : np.ndarray
        Q (quadrature) data for ground state.
    Ie : np.ndarray
        I (in-phase) data for excited state.
    Qe : np.ndarray
        Q (quadrature) data for excited state.

    Returns
    -------
    dict
        Dictionary with key 'theta' containing the calculated rotation angle in radians.
    """
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    theta = -np.arctan2((ye - yg), (xe - xg))
    return {"theta": theta}


def fit_ge_by_center(signals: np.ndarray, **kwargs) -> tuple:
    """
    Analyze ground and excited state signals using the center method.

    This is a wrapper around fitting_ge_and_plot that uses the center method
    to determine the optimal rotation angle for state discrimination.

    Parameters
    ----------
    signals : np.ndarray
        Complex array of shape (2, N) containing measurement signals.
        First row should contain ground state signals, second row excited state signals.

    Returns
    -------
    tuple
        A tuple containing:
        - fidelity: The assignment fidelity between ground and excited states (0.5-1.0)
        - threshold: The optimal threshold value for state discrimination
        - theta_deg: The optimal rotation angle in degrees
    """
    return fitting_ge_and_plot(signals, get_rotate_angle, **kwargs)
