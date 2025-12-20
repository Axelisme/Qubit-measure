import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

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
    X_integrated = cumulative_trapezoid(populations, times, axis=0, initial=0)
    P0 = populations[0]
    Y_diff = populations - P0

    solution, *_ = np.linalg.lstsq(X_integrated, Y_diff, rcond=None)
    M_estimated = solution.T

    diagonals = np.diag(M_estimated).copy()
    M_estimated[M_estimated < 0] = 0
    np.fill_diagonal(M_estimated, diagonals)
    for i in range(3):
        M_estimated[i, i] = -(np.sum(M_estimated[:, i]) - M_estimated[i, i])

    T_ge = M_estimated[1, 0]
    T_eg = M_estimated[0, 1]
    T_eo = M_estimated[2, 1]
    T_oe = M_estimated[1, 2]
    T_go = M_estimated[0, 2]
    T_og = M_estimated[2, 0]
    pg0 = P0[0]
    pe0 = P0[1]

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
