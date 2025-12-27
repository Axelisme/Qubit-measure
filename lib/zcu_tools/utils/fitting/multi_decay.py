import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter1d

from .base import fit_func


def model_func(t, T_ge, T_eg, T_eo, T_oe, T_go, T_og, pg0, pe0) -> NDArray[np.float64]:
    if pg0 + pe0 > 1.0:
        p_sum = pg0 + pe0
        pg0 /= p_sum
        pe0 /= p_sum

    P0 = np.array([pg0, pe0, 1.0 - pg0 - pe0])
    M = np.array(
        [
            [-(T_ge + T_go), T_eg, T_og],
            [T_ge, -(T_eg + T_eo), T_oe],
            [T_go, T_eo, -(T_og + T_oe)],
        ]
    )

    # P(t) = exp(Mt) P(0) = V exp(Dt) V^-1 P(0)
    evals, evecs = np.linalg.eig(M)
    W = evecs * np.linalg.solve(evecs, P0)
    exp_terms = np.exp(np.outer(t, evals))
    P_t = exp_terms @ W.T

    return np.real(P_t)


def guess_initial_params(
    populations: NDArray[np.float64], times: NDArray[np.float64]
) -> Tuple[float, ...]:
    """
    使用積分法最小平方估計三能階系統速率常數。
    假設 populations 欄位順序為: [Ground (g), Excited (e), Other (o)]
    """
    # 1. 數值積分: \int P(dt)
    P_int = cumulative_trapezoid(populations, times, axis=0, initial=0)

    # 2. 差分: P(t) - P(0)
    P0 = populations[0]
    delta_P = populations - P0

    # 初始機率分布
    pg0 = max(0.0, min(1.0, P0[0]))
    pe0 = max(0.0, min(1.0, P0[1]))

    # Variable order in x: [T_ge, T_eg, T_eo, T_oe, T_go, T_og]
    # Index mapping:        0     1     2     3     4     5

    Ig = P_int[:, 0]
    Ie = P_int[:, 1]
    Io = P_int[:, 2]

    # Zeros array for padding
    Z = np.zeros_like(Ig)

    # Construct Matrix A for Ax = b
    # Equation for g: Pg - Pg0 = -Ig(T_ge + T_go) + Ie(T_eg) + Io(T_og)
    # x coeffs: [-Ig, Ie, 0, 0, -Ig, Io]
    A_g = np.stack([-Ig, Ie, Z, Z, -Ig, Io], axis=1)
    b_g = delta_P[:, 0]

    # Equation for e: Pe - Pe0 = Ig(T_ge) + Io(T_oe) - Ie(T_eg + T_eo)
    # x coeffs: [Ig, -Ie, -Ie, Io, 0, 0]
    A_e = np.stack([Ig, -Ie, -Ie, Io, Z, Z], axis=1)
    b_e = delta_P[:, 1]

    # Equation for o: Po - Po0 = Ig(T_go) + Ie(T_eo) - Io(T_og + T_oe)
    # x coeffs: [Z, Z, Ie, -Io, Ig, -Io]
    A_o = np.stack([Z, Z, Ie, -Io, Ig, -Io], axis=1)
    b_o = delta_P[:, 2]

    # Stack all equations
    A = np.vstack([A_g, A_e, A_o])
    b = np.concatenate([b_g, b_e, b_o])

    # Solve Ax = b subject to x >= 0
    x, _ = nnls(A, b)

    T_ge, T_eg, T_eo, T_oe, T_go, T_og = x

    return (T_ge, T_eg, T_eo, T_oe, T_go, T_og, pg0, pe0)


def fit_transition_rates(
    times: NDArray[np.float64],
    populations: NDArray[np.float64],
    p0_guess: Optional[Tuple[float, ...]] = None,
) -> Tuple[Tuple[float, ...], NDArray[np.float64], Tuple[float, ...]]:
    """fitted_rates: (T_ge, T_eg, T_eo, T_oe, T_go, T_og)"""

    populations = gaussian_filter1d(populations, sigma=1, axis=0)

    if p0_guess is None:
        p0_guess = guess_initial_params(populations, times)

    R_ge, R_eg, R_eo, R_oe, R_go, R_og, p0_g, p0_e = p0_guess

    max_R = 2 * np.max([R_ge, R_eg, R_eo, R_oe, R_go, R_og])
    bounds = (
        [0, 0, 0, 0, 0, 0, max(0, p0_g - 0.1), max(0, p0_e - 0.1)],
        [
            max_R,
            max_R,
            max_R,
            max_R,
            max_R,
            max_R,
            min(1, p0_g + 0.01),
            max(1, p0_e + 0.01),
        ],
    )

    pOpt, _ = fit_func(
        times,
        populations.flatten(),
        lambda *args: model_func(*args).flatten(),
        p0_guess,
        bounds=bounds,
    )

    fit_populations = model_func(times, *pOpt)

    R_ge, R_eg, R_eo, R_oe, R_go, R_og, *_ = pOpt

    fitted_rates = (R_ge, R_eg, R_eo, R_oe, R_go, R_og)

    return fitted_rates, fit_populations, tuple(pOpt)
