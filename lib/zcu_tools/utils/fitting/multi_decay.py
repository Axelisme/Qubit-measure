import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import nnls

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

    # 2. 差分矩陣: P(t) - P(0)
    # 為了減少雜訊影響，P0 也可以考慮用平均值或回歸預測，這裡維持邏輯簡潔
    P0 = populations[0]
    delta_P = populations - P0

    # 初始機率分布 (直接從數據獲取)
    pg0 = max(0.0, min(1.0, P0[0]))
    pe0 = max(0.0, min(1.0, P0[1]))

    # fit T_eg, T_og
    dP_int_xg = np.stack(
        [P_int[:, 1] - P_int[:, 0], P_int[:, 2] - P_int[:, 0]], axis=-1
    )
    (T_eg, T_og), *_ = nnls(dP_int_xg, delta_P[:, 0])

    # fit T_ge, T_oe
    dP_int_xe = np.stack(
        [P_int[:, 0] - P_int[:, 1], P_int[:, 2] - P_int[:, 1]], axis=-1
    )
    (T_ge, T_oe), *_ = nnls(dP_int_xe, delta_P[:, 1])

    # fit T_go, T_eo
    dP_int_xo = np.stack(
        [P_int[:, 0] - P_int[:, 2], P_int[:, 1] - P_int[:, 2]], axis=-1
    )
    (T_go, T_eo), *_ = nnls(dP_int_xo, delta_P[:, 2])

    return (T_ge, T_eg, T_eo, T_oe, T_go, T_og, pg0, pe0)


def fit_transition_rates(
    times: NDArray[np.float64],
    populations: NDArray[np.float64],
    p0_guess: Optional[Tuple[float, ...]] = None,
) -> Tuple[Tuple[float, ...], NDArray[np.float64], Tuple[float, ...]]:
    """fitted_rates: (T_ge, T_eg, T_eo, T_oe, T_go, T_og)"""

    if p0_guess is None:
        p0_guess = guess_initial_params(populations, times)

    R_ge, R_eg, R_eo, R_oe, R_go, R_og, *_ = p0_guess

    max_R = 2 * np.max([R_ge, R_eg, R_eo, R_oe, R_go, R_og])
    bounds = (
        [0, 0, 0, 0, 0, 0, 0, 0],
        [max_R, max_R, max_R, max_R, max_R, max_R, 1, 1],
    )

    # pOpt, _ = fit_func(
    #     times,
    #     populations.flatten(),
    #     lambda *args: model_func(*args).flatten(),
    #     p0_guess,
    #     bounds=bounds,
    # )
    pOpt = p0_guess

    fit_populations = model_func(times, *pOpt)

    R_ge, R_eg, R_eo, R_oe, R_go, R_og, *_ = pOpt

    fitted_rates = (R_ge, R_eg, R_eo, R_oe, R_go, R_og)

    return fitted_rates, fit_populations, tuple(pOpt)
