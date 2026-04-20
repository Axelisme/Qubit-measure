from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Optional, cast

from .base import cosfunc, decaycos, fitcos, fitdecaycos


def fit_rabi(
    xs: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    /,
    decay: bool = False,
    init_phase: Optional[float] = None,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    NDArray[np.float64],
    tuple[tuple[float, ...], NDArray[np.float64]],
]:
    """Return (pi_x, pi_x_err, pi2_x, pi2_x_err, freq, freq_err, fit_signals, (pOpt, pCov))"""

    # choose fitting function
    fixedparams: list[Optional[float]]
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

    var_freq = float(pCov[2, 2])
    var_phase = float(pCov[3, 3])
    cov_pf = float(pCov[3, 2])
    freq_err = float(np.sqrt(max(var_freq, 0.0)))

    # derive pi / pi/2 positions from phase
    if phase > 270:
        A_pi, A_pi2 = 1.5, 1.25
    elif phase < 90:
        A_pi, A_pi2 = 0.5, 0.25
    else:
        A_pi, A_pi2 = 1.0, 0.75

    pi_x = (A_pi - phase / 360) / freq
    pi2_x = (A_pi2 - phase / 360) / freq

    # error propagation: x = (A - phase/360) / freq
    # dx/dphase = -1/(360*freq),  dx/dfreq = -x/freq
    def _xerr(x: float) -> float:
        dphase = -1.0 / (360.0 * freq)
        dfreq = -x / freq
        var = dphase**2 * var_phase + dfreq**2 * var_freq + 2 * dphase * dfreq * cov_pf
        return float(np.sqrt(max(var, 0.0)))

    pi_x_err = _xerr(pi_x)
    pi2_x_err = _xerr(pi2_x)

    # while pi2_x < min_length:
    #     pi2_x += 1.0 / freq
    #     pi_x += 1.0 / freq

    pOpt = cast(tuple[float, float, float, float, float], tuple(pOpt))

    return (
        float(pi_x),
        pi_x_err,
        float(pi2_x),
        pi2_x_err,
        float(freq),
        freq_err,
        fit_signals,
        (pOpt, pCov),
    )
