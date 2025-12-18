from typing import Literal, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .base import fitlor, fitsinc, lorfunc, sincfunc


def fit_qubit_freq(
    fpts: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    type: Literal["lor", "sinc"] = "lor",
) -> Tuple[
    float,
    float,
    float,
    float,
    NDArray[np.float64],
    Tuple[Tuple[float, ...], NDArray[np.float64]],
]:
    """[freq, freq_err, kappa, kappa_err, fit_singals, (pOpt, pCov)]"""
    if type == "lor":
        pOpt, pCov = fitlor(fpts, real_signals)
        fit_singals = lorfunc(fpts, *pOpt)

        freq = pOpt[3]
        kappa = 2 * pOpt[4]
        freq_err = np.sqrt(np.diag(pCov))[3]
        kappa_err = 2 * np.sqrt(np.diag(pCov))[4]

    elif type == "sinc":
        pOpt, pCov = fitsinc(fpts, real_signals)
        fit_singals = sincfunc(fpts, *pOpt)

        freq = pOpt[3]
        kappa = 1.2067 * pOpt[4]  # sinc function hwm is 1.2067 * gamma
        freq_err = np.sqrt(np.diag(pCov))[3]
        kappa_err = 1.2067 * np.sqrt(np.diag(pCov))[4]

    pOpt = cast(Tuple[float, float, float, float, float], tuple(pOpt))

    return freq, freq_err, kappa, kappa_err, fit_singals, (pOpt, pCov)
