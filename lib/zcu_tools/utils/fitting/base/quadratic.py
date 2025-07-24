from typing import Tuple

import numpy as np


def quadratic_fit(
    xs: np.ndarray, ys: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """
    Fit a quadratic function to the data.

    Use equation:
        ax^2 + bx + cy^2 + dy + exy + f = 0

    Args:
        xs: x coordinates of the data
        ys: y coordinates of the data

    Returns:
        a, b, c, d, e, f: coefficients of the quadratic function
    """
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not mask.any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    xs = xs[mask]
    ys = ys[mask]

    A = np.column_stack([xs**2, xs, ys**2, ys, xs * ys, np.ones_like(xs)])
    U, S, Vh = np.linalg.svd(A)
    V = Vh[-1, :]
    a, b, c, d, e, f = V / np.linalg.norm(V)
    return a, b, c, d, e, f


def quadratic_fit_wo_a(
    xs: np.ndarray, ys: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Fit a modified quadratic function to the data.

    Use equation:
        bx + cy^2 + dy + exy + f = 0

    Args:
        xs: x coordinates of the data
        ys: y coordinates of the data

    Returns:
        b, c, d, e, f: coefficients of the quadratic function
    """
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not mask.any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    xs = xs[mask]
    ys = ys[mask]

    A = np.column_stack([xs, ys**2, ys, xs * ys, np.ones_like(xs)])
    U, S, Vh = np.linalg.svd(A)
    V = Vh[-1, :]
    b, c, d, e, f = V / np.linalg.norm(V)
    return b, c, d, e, f


def get_asymptotes(
    a: float, b: float, c: float, d: float, e: float, f: float
) -> Tuple[float, float]:
    """
    Get the slopes of the asymptotes of the quadratic equation.

    Returns:
        m1, m2: slopes of the two asymptotes (y = m x)
    """
    # Solve c m^2 + e m + a = 0
    if c == 0:
        # Avoid division by zero
        if e == 0:
            # a = 0: degenerate case
            return (np.nan, np.nan)
        m1 = -a / e
        m2 = m1
    else:
        discrim = e**2 - 4 * c * a
        if discrim < 0:
            # No real asymptotes
            return (np.nan, np.nan)
        sqrt_discrim = np.sqrt(discrim)
        m1 = (-e + sqrt_discrim) / (2 * c)
        m2 = (-e - sqrt_discrim) / (2 * c)
    return m1, m2
