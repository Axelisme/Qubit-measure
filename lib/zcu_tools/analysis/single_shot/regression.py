import numpy as np

from .base import fitting_ge_and_plot


def get_rotate_angle(
    Ig: np.ndarray, Qg: np.ndarray, Ie: np.ndarray, Qe: np.ndarray
) -> dict:
    """
    Calculate the optimal rotation angle using logistic regression.

    This method uses logistic regression to find the optimal decision boundary
    between ground and excited state clusters, then calculates the rotation angle
    to align this boundary with the I-axis.

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
        Dictionary containing:
        - 'theta': Calculated rotation angle in radians
        - 'threshold': Optimal threshold value from logistic regression (not used in base workflow)
    """
    # use logictic regression to get the rotate angle
    from sklearn.linear_model import LogisticRegression

    X = np.vstack((np.column_stack((Ig, Qg)), np.column_stack((Ie, Qe))))
    Y = np.hstack((np.zeros(len(Ig)), np.ones(len(Ie))))

    model = LogisticRegression(
        penalty="l1", C=100, solver="liblinear", max_iter=1000
    ).fit(X, Y)

    # ax + by + c = 0
    a, b = model.coef_[0]
    c = model.intercept_[0]

    theta = -np.arctan2(b, a)
    threshold = -c / np.sqrt(a**2 + b**2)

    return {"theta": theta, "threshold": threshold}


def fit_ge_by_regression(signals: np.ndarray, plot: bool = True) -> tuple:
    """
    Analyze ground and excited state signals using logistic regression.

    This is a wrapper around fitting_ge_and_plot that uses logistic regression
    to determine the optimal rotation angle for state discrimination.

    Parameters
    ----------
    signals : np.ndarray
        Complex array of shape (2, N) containing measurement signals.
        First row should contain ground state signals, second row excited state signals.
    plot : bool, default=True
        If True, generate visualization plots of the analysis results.

    Returns
    -------
    tuple
        A tuple containing:
        - fidelity: The assignment fidelity between ground and excited states (0.5-1.0)
        - threshold: The optimal threshold value for state discrimination
        - theta_deg: The optimal rotation angle in degrees
    """
    return fitting_ge_and_plot(signals, get_rotate_angle, plot)
