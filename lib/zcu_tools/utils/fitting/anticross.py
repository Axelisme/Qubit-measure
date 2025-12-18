from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize
from tqdm.auto import tqdm

from .base import (
    encode_params,
    quadratic_fit,
    quadratic_fit_wo_a,
    retrieve_params,
)


def get_predict_ys(
    xs: NDArray[np.float64], a: float, b: float, c: float, d: float, e: float, f: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    xs: NDArray[np.float64],
    ys1: NDArray[np.float64],
    ys2: NDArray[np.float64],
    horizontal_line: bool = False,
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
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # ---------------------------------------------------------------------
    # 2. Define residuals for least-squares optimisation
    # ---------------------------------------------------------------------
    def _residual(params: NDArray[np.float64]) -> NDArray[np.float64]:
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
    xs: NDArray[np.float64],
    fpts1: NDArray[np.float64],
    fpts2: NDArray[np.float64],
    horizontal_line: bool = False,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    NDArray[np.float64],
    NDArray[np.float64],
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
    params = fit_hyperbolic(xs, fpts1, fpts2, horizontal_line=horizontal_line)

    fit_fpts1, fit_fpts2 = get_predict_ys(xs, *params)

    return *encode_params(*params), fit_fpts1, fit_fpts2, params


def fit_anticross2d(
    xs: NDArray[np.float64], fpts: NDArray[np.float64], signals: NDArray[np.complex128]
) -> Tuple[float, float, float, float, float]:
    MAX_ITER = 1000

    pbar = tqdm(total=MAX_ITER, desc="Auto fitting", leave=False)

    def update_pbar(cx, cy, width) -> None:
        pbar.update(1)
        pbar.set_postfix_str(f"({cx:.3f}, {cy:.3f}, {width:.3f})")

    amps = np.abs(signals).astype(np.float64)

    # determine whether fit the value to max or min
    if np.sum(amps[:, amps.shape[1] // 2] - amps[:, 0]) > 0:
        amps = np.max(amps) - amps  # make peak always is the maximum

    # guess the initial parameters
    cx = np.sum(xs[:, None] * amps) / np.sum(amps)
    cy = fpts[np.argmin(np.max(amps, axis=1))]
    width = (fpts.max() - fpts.min()) / 100
    m1 = 0.0
    m2 = 1.0

    # derive the fitting tolerance
    ftol = np.max(amps) * 1e-4

    def loss_fn(cx, cy, width, m1, m2) -> float:
        update_pbar(cx, cy, width)

        fit_fpts1, fit_fpts2 = get_predict_ys(
            xs, *retrieve_params(cx, cy, width, m1, m2)
        )

        # 用線性插值取得每個 rf_0 對應的 signal
        vals1 = [np.interp(f, fpts, amps[i]) for i, f in enumerate(fit_fpts1)]
        vals2 = [np.interp(f, fpts, amps[i]) for i, f in enumerate(fit_fpts2)]
        return float(-np.nanmean(vals1) - np.nanmean(vals2))

    fit_kwargs = dict(
        method="L-BFGS-B",
        options={"disp": False, "maxiter": MAX_ITER, "ftol": ftol},
    )

    res = minimize(
        lambda p: loss_fn(*p),
        x0=[cx, cy, width, m1, m2],
        **fit_kwargs,
    )
    if not isinstance(res, np.ndarray):
        res = res.x  # compatibility with scipy < 1.7

    cx, cy, width, m1, m2 = res

    return cx, cy, width, m1, m2
