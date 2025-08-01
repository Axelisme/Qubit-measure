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


def encode_params(
    a: float, b: float, c: float, d: float, e: float, f: float
) -> Tuple[float, float, float, float, float]:
    """
    Encode the parameters of the quadratic function into some anti-crossing properties

    Args:
        a, b, c, d, e, f: coefficients of the quadratic function

    Returns:
        cx: center x
        cy: center y
        width: width of the anticrossing
        m1: slope of the first asymptote
        m2: slope of the second asymptote
    """
    # 求中心點 (vertex of the conic)
    discrim = e**2 - 4 * a * c

    if discrim == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    cx = (2 * b * c - e * d) / discrim
    cy = (2 * a * d - e * b) / discrim

    # Solve c m^2 + e m + a = 0
    if c == 0:
        m1 = -a / e if e != 0 else np.nan
        m2 = m1
    else:
        m1 = (-e + np.sqrt(discrim)) / (2 * c)
        m2 = (-e - np.sqrt(discrim)) / (2 * c)

    A = c
    B = d + e * cx
    C = a * cx**2 + b * cx + f
    D = B**2 - 4 * A * C

    if D < 0:
        return cx, cy, np.nan, np.nan, np.nan

    width = 0.5 * np.sqrt(D) / np.abs(A)

    return cx, cy, width, m1, m2


def retrieve_params(
    cx: float, cy: float, width: float, m1: float, m2: float
) -> Tuple[float, float, float, float, float, float]:
    """
    Retrieve the parameters from some anti-crossing properties

    Args:
        cx: center x
        cy: center y
        width: width of the anticrossing
        m1: slope of the first asymptote
        m2: slope of the second asymptote

    Returns:
        a, b, c, d, e, f: coefficients of the quadratic function
    """
    # Validate inputs
    if any(np.isnan([cx, cy, width, m1, m2])):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Width must be positive and the two slopes must be different (otherwise discrim = 0)
    if width <= 0 or np.isclose(m1, m2):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Choose a convenient scaling: set c = 1. Scaling does not affect the encoded parameters.
    c = 1.0

    # Recover e and a from the asymptote slopes m1, m2 using
    #     c m^2 + e m + a = 0  →  m1 + m2 = -e/c ,  m1 m2 = a/c
    e = -(m1 + m2)
    a = m1 * m2

    # Recover b and d from the center (cx, cy)
    # Using the relations derived from encode_params:
    #     cx * delta = 2 b - e d
    #     cy * delta = 2 a d - e b
    # Solve the 2×2 linear system for b and d
    b = -(2 * a * cx + e * cy)
    d = -(2 * cy + e * cx)

    # Recover f from the width definition. In encode_params:
    #     A = c (which we set to 1)
    #     B = d + e * cx
    #     C = a * cx**2 + b * cx + f
    #     width = 0.5 * sqrt(B**2 - 4 * A * C) / |A|
    # →  (d + e * cx)^2 - 4 (a * cx**2 + b * cx + f) = (2 * width)^2
    B = d + e * cx
    f = -(width**2) + (B**2) / 4 - a * cx**2 - b * cx

    return a, b, c, d, e, f
