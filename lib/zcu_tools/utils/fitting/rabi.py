from typing import Tuple

import numpy as np

from .base import cosfunc, decaycos, fitcos, fitdecaycos


def fit_rabi(
    xs: np.ndarray, real_signals: np.ndarray, *, decay: bool = False
) -> Tuple[float, float, np.ndarray, Tuple[Tuple[float, ...], np.ndarray]]:
    """Return (pi_x, pi2_x)"""

    # choose fitting function
    fit_func = fitdecaycos if decay else fitcos
    cos_func = decaycos if decay else cosfunc

    pOpt, pCov = fit_func(xs, real_signals)

    fit_signals = cos_func(xs, *pOpt)

    freq: float = pOpt[2]
    phase: float = pOpt[3] % 360

    # derive pi / pi/2 positions from phase
    if phase > 270:
        pi_x = (1.5 - phase / 360) / freq
        pi2_x = (1.25 - phase / 360) / freq
    elif phase < 90:
        pi_x = (0.5 - phase / 360) / freq
        pi2_x = (0.25 - phase / 360) / freq
    else:
        pi_x = (1.0 - phase / 360) / freq
        pi2_x = (0.75 - phase / 360) / freq

    return float(pi_x), float(pi2_x), fit_signals, (pOpt, pCov)
