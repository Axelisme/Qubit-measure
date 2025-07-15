from typing import Tuple

import numpy as np

from .base import decaycos, dual_expfunc, expfunc, fit_dualexp, fitdecaycos, fitexp


def fit_decay(
    xs: np.ndarray, real_signals: np.ndarray
) -> Tuple[float, float, np.ndarray, Tuple[Tuple[float, ...], np.ndarray]]:
    pOpt, pCov = fitexp(xs, real_signals)

    fit_signals = expfunc(xs, *pOpt)

    t1: float = pOpt[2]
    t1err: float = np.sqrt(pCov[2, 2])
    return t1, t1err, fit_signals, (pOpt, pCov)


def fit_dual_decay(
    xs: np.ndarray, real_signals: np.ndarray
) -> Tuple[
    float,
    ...,
    np.ndarray,
    Tuple[Tuple[float, ...], np.ndarray],
]:
    pOpt, pCov = fit_dualexp(xs, real_signals)

    # make sure t1 is the longer one
    if pOpt[4] > pOpt[2]:
        pOpt = [pOpt[0], pOpt[3], pOpt[4], pOpt[1], pOpt[2]]
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


def fit_decay_fringe(
    xs: np.ndarray, real_signals: np.ndarray
) -> Tuple[float, ..., np.ndarray, Tuple[Tuple[float, ...], np.ndarray]]:
    pOpt, pCov = fitdecaycos(xs, real_signals)

    fit_signals = decaycos(xs, *pOpt)

    t2f: float = pOpt[4]
    t2ferr: float = np.sqrt(pCov[4, 4])
    detune: float = pOpt[2]
    detune_err: float = np.sqrt(pCov[2, 2])

    return t2f, t2ferr, detune, detune_err, fit_signals, (pOpt, pCov)
