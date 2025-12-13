from typing import List, Optional, Tuple, cast

import numpy as np

from .base import cosfunc, decaycos, fitcos, fitdecaycos


def fit_rabi(
    xs: np.ndarray,
    real_signals: np.ndarray,
    *,
    decay: bool = False,
    init_phase: Optional[float] = None,
    min_length: float = 0.0,
) -> Tuple[float, float, float, np.ndarray, Tuple[Tuple[float, ...], np.ndarray]]:
    """Return (pi_x, pi2_x, freq, fit_signals, (pOpt, pCov))"""

    # choose fitting function
    fixedparams: List[Optional[float]]
    if decay:
        fit_func = fitdecaycos
        cos_func = decaycos
        fixedparams = [None] * 5
        fixedparams[3] = init_phase
    else:
        fit_func = fitcos
        cos_func = cosfunc
        fixedparams = [None] * 4
        fixedparams[3] = init_phase

    pOpt, pCov = fit_func(xs, real_signals, fixedparams=fixedparams)

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

    while pi2_x < min_length:
        pi2_x += 0.5 / freq
        pi_x += 0.5 / freq

    pOpt = cast(Tuple[float, float, float, float, float], tuple(pOpt))

    return float(pi_x), float(pi2_x), float(freq), fit_signals, (pOpt, pCov)
