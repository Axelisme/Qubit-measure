from typing import Optional, Tuple

import numpy as np

from .base import (
    batch_fit_func,
    decaycos,
    dual_expfunc,
    expfunc,
    fit_dualexp,
    fit_gauss,
    fitdecaycos,
    fitexp,
    gauss_func,
)


def fit_decay(
    xs: np.ndarray,
    real_signals: np.ndarray,
    fit_params: Optional[Tuple[float, float, float]] = None,
) -> Tuple[float, float, np.ndarray, Tuple[Tuple[float, ...], np.ndarray]]:
    """return [t1, t1err, fit_signals, (pOpt, pCov)]"""
    pOpt, pCov = fitexp(xs, real_signals, fitparams=fit_params)

    fit_signals = expfunc(xs, *pOpt)

    t1: float = pOpt[2]
    t1err: float = np.sqrt(pCov[2, 2])
    return t1, t1err, fit_signals, (pOpt, pCov)


def fit_dual_decay(
    xs: np.ndarray, real_signals: np.ndarray
) -> Tuple[
    float,
    float,
    float,
    float,
    np.ndarray,
    Tuple[Tuple[float, ...], np.ndarray],
]:
    """return [t1, t1err, t1b, t1berr, fit_signals, (pOpt, pCov)]"""
    pOpt, pCov = fit_dualexp(xs, real_signals)

    # make sure t1 is the longer one
    if pOpt[4] > pOpt[2]:
        pOpt = (pOpt[0], pOpt[3], pOpt[4], pOpt[1], pOpt[2])
        new_pCov = pCov.copy()
        new_pCov[[1, 3], :] = new_pCov[[3, 1], :]
        new_pCov[[2, 4], :] = new_pCov[[4, 2], :]
        new_pCov[:, [1, 3]] = new_pCov[:, [3, 1]]
        new_pCov[:, [2, 4]] = new_pCov[:, [4, 2]]
        pCov = new_pCov

    fit_signals = dual_expfunc(xs, *pOpt)

    t1: float = pOpt[2]
    t1err: float = np.sqrt(pCov[2, 2])
    t1b: float = pOpt[4]
    t1berr: float = np.sqrt(pCov[4, 4])

    return t1, t1err, t1b, t1berr, fit_signals, (pOpt, pCov)


def fit_ge_decay(
    times: np.ndarray,
    g_signals: np.ndarray,
    e_signals: np.ndarray,
    share_t1: bool = True,
) -> Tuple[Tuple[float, float, np.ndarray], Tuple[float, float, np.ndarray]]:
    """return [(g_t1, g_t1err, g_fit_signals), (e_t1, e_t1err, e_fit_signals)]"""
    g_init_p, _ = fitexp(times, g_signals)  # (y0, yscale, decay)
    e_init_p, _ = fitexp(times, e_signals)  # (y0, yscale, decay)

    shared_idxs = [0]
    if share_t1:
        shared_idxs.append(2)

    ge_params, ge_pcov = batch_fit_func(
        [times, times],
        [g_signals, e_signals],
        expfunc,
        [g_init_p, e_init_p],
        shared_idxs,
    )
    g_params, e_params = ge_params
    g_pCov, e_pCov = ge_pcov

    g_t1 = g_params[2]
    g_t1err = np.sqrt(g_pCov[2, 2])
    e_t1 = e_params[2]
    e_t1err = np.sqrt(e_pCov[2, 2])

    g_fit_signals = expfunc(times, *g_params)
    e_fit_signals = expfunc(times, *e_params)

    return (g_t1, g_t1err, g_fit_signals), (e_t1, e_t1err, e_fit_signals)


def fit_decay_fringe(
    xs: np.ndarray,
    real_signals: np.ndarray,
    fit_params: Optional[Tuple[float, ...]] = None,
) -> Tuple[
    float, float, float, float, np.ndarray, Tuple[Tuple[float, ...], np.ndarray]
]:
    """return [t2f, t2ferr, detune, detune_err, fit_signals, (pOpt, pCov)]"""
    pOpt, pCov = fitdecaycos(xs, real_signals, fitparams=fit_params)

    fit_signals = decaycos(xs, *pOpt)

    t2f: float = pOpt[4]
    t2ferr: float = np.sqrt(pCov[4, 4])
    detune: float = pOpt[2]
    detune_err: float = np.sqrt(pCov[2, 2])

    return t2f, t2ferr, detune, detune_err, fit_signals, (pOpt, pCov)


def fit_gauss_decay(
    xs: np.ndarray,
    real_signals: np.ndarray,
    fit_params: Optional[Tuple[float, ...]] = None,
) -> Tuple[float, float, np.ndarray, Tuple[Tuple[float, ...], np.ndarray]]:
    """return [t2g, t2gerr, fit_signals, (pOpt, pCov)]"""
    pOpt, pCov = fit_gauss(
        xs, real_signals, fitparams=fit_params, fixedparams=[None, None, 0.0, None]
    )

    fit_signals = gauss_func(xs, *pOpt)

    sigma: float = pOpt[2]
    sigma_err: float = np.sqrt(pCov[2, 2])

    # effective T2
    # f(0) * exp(-1) = f(0) * exp(-t2^2 / (2*sigma^2))
    t2 = np.sqrt(2) * sigma
    t2_err = np.sqrt(2) * sigma_err

    return t2, t2_err, fit_signals, (pOpt, pCov)
