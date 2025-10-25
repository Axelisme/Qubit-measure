from typing import Literal, Tuple

import numpy as np

from .base import fitlor, fitsinc, lorfunc, sincfunc


def fit_qubit_freq(
    fpts: np.ndarray, real_signals: np.ndarray, type: Literal["lor", "sinc"] = "lor"
) -> Tuple[
    float, float, float, float, np.ndarray, Tuple[Tuple[float, ...], np.ndarray]
]:
    """[freq, freq_err, kappa, kappa_err, fit_singals, (pOpt, pCov)]"""
    if type == "lor":
        pOpt, pCov = fitlor(fpts, real_signals)
        fit_singals = lorfunc(fpts, *pOpt)

        freq: float = pOpt[3]
        kappa: float = 2 * pOpt[4]
        freq_err = np.sqrt(np.diag(pCov))[3]
        kappa_err = 2 * np.sqrt(np.diag(pCov))[4]

    elif type == "sinc":
        pOpt, pCov = fitsinc(fpts, real_signals)
        fit_singals = sincfunc(fpts, *pOpt)

        freq: float = pOpt[3]
        kappa: float = 1.2067 * pOpt[4]  # sinc function hwm is 1.2067 * gamma
        freq_err = np.sqrt(np.diag(pCov))[3]
        kappa_err = 1.2067 * np.sqrt(np.diag(pCov))[4]

    return freq, freq_err, kappa, kappa_err, fit_singals, (pOpt, pCov)
