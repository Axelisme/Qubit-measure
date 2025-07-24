from typing import Tuple

import numpy as np
from scipy.optimize import least_squares

from .base import get_asymptotes, quadratic_fit, quadratic_fit_wo_a


def get_predict_ys(
    xs: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float
) -> Tuple[np.ndarray, np.ndarray]:
    # 對每個 xs 求兩個 y，分別對應 ys1, ys2, 其中 ys1 > ys2

    # c y^2 + (d + e x) y + (a x^2 + b x + f) = 0
    A = c
    B = d + e * xs
    C = a * xs**2 + b * xs + f
    discrim = B**2 - 4 * A * C
    discrim[discrim < 0] = np.nan

    ys1 = (-B + np.sqrt(discrim)) / (2 * A)
    ys2 = (-B - np.sqrt(discrim)) / (2 * A)

    if A < 0:
        ys1, ys2 = ys2, ys1

    return ys1, ys2


def fit_hyperbolic(
    xs: np.ndarray, ys1: np.ndarray, ys2: np.ndarray, horizontal_line: bool = False
) -> Tuple[float, float, float, float, float, float]:
    if np.sum(ys1 > ys2) < len(ys1) / 2:
        ys1, ys2 = ys2, ys1

    # ---------------------------------------------------------------------
    # 1. Build initial guess from a simple quadratic fit
    # ---------------------------------------------------------------------
    if horizontal_line:
        a = 0.0
        b, c, d, e, f = quadratic_fit_wo_a(
            np.concatenate((xs, xs)),  # x-values for both branches
            np.concatenate((ys1, ys2)),  # corresponding y-values
        )
    else:
        a, b, c, d, e, f = quadratic_fit(
            np.concatenate((xs, xs)),  # x-values for both branches
            np.concatenate((ys1, ys2)),  # corresponding y-values
        )

    if np.any(np.isnan([a, b, c, d, e, f])):
        return (np.nan,) * 6

    # ---------------------------------------------------------------------
    # 2. Define residuals for least-squares optimisation
    # ---------------------------------------------------------------------
    def _residual(params: np.ndarray) -> np.ndarray:
        if horizontal_line:
            a = 0.0
            b, c, d, e, f = params
        else:
            a, b, c, d, e, f = params

        # Predicted branches for all xs.
        y1_pred, y2_pred = get_predict_ys(xs, a, b, c, d, e, f)

        # Residuals for upper and lower branches.  Ignore NaN predictions or
        # measurements so that they do not contaminate the optimisation.
        res_upper = y1_pred - ys1
        res_lower = y2_pred - ys2

        res_upper[np.isnan(res_upper)] = np.inf
        res_lower[np.isnan(res_lower)] = np.inf

        residuals = np.concatenate([res_upper, res_lower])

        return residuals

    # ---------------------------------------------------------------------
    # 3. Run optimisation.  Bounds are left unconstrained because the initial
    #    guess is usually close to the optimum; users can post-process if
    #    required.
    # ---------------------------------------------------------------------
    if horizontal_line:
        init_params = np.array([b, c, d, e, f])
    else:
        init_params = np.array([a, b, c, d, e, f])

    lsq_result = least_squares(_residual, init_params, method="lm")

    if not lsq_result.success:
        return a, b, c, d, e, f

    if horizontal_line:
        fit_params = np.array([0.0, *lsq_result.x])
    else:
        fit_params = lsq_result.x

    return tuple(fit_params / np.linalg.norm(fit_params))


def fit_anticross(
    xs: np.ndarray, fpts1: np.ndarray, fpts2: np.ndarray, horizontal_line: bool = False
) -> Tuple[
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    Tuple[float, float, float, float, float, float],
]:
    """
    Fit a anticrossing pattern to the data.

    Args:
        xs: x coordinates of the data
        fpts1: fpts of the first line
        fpts2: fpts of the second line


    Returns:
        center_x, center_y, g, m1, m2, fit_fpts1, fit_fpts2, params:
    """

    # ax^2 + bx + cy^2 + dy + exy + f = 0
    a, b, c, d, e, f = fit_hyperbolic(xs, fpts1, fpts2, horizontal_line=horizontal_line)

    # 求中心點 (vertex of the conic)
    D = e**2 - 4 * a * c
    if D == 0:
        # parallel
        cx = np.mean(xs)
        cy = 0.5 * (np.mean(fpts1) + np.mean(fpts2))
    else:
        cx = (2 * b * c - e * d) / D
        cy = (2 * a * d - e * b) / D

    # 計算 g: 在 center_x 處，兩根的距離
    A = c
    B = d + e * cx
    C = a * cx**2 + b * cx + f
    discrim = B**2 - 4 * A * C
    if discrim >= 0:
        g01 = 0.5 * np.sqrt(discrim) / np.abs(A)
    else:
        g01 = 0.5 * np.abs(np.mean(fpts2) - np.mean(fpts1))

    m1, m2 = get_asymptotes(a, b, c, d, e, f)

    fit_fpts1, fit_fpts2 = get_predict_ys(xs, a, b, c, d, e, f)

    return cx, cy, g01, m1, m2, fit_fpts1, fit_fpts2, (a, b, c, d, e, f)
