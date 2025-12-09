from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from .base import assign_init_p, fit_func


# exponential decay function
def expfunc(x: np.ndarray, *p: float) -> np.ndarray:
    """p = [y0, yscale, decay_time]"""
    y0, yscale, decay_time = p
    return y0 + yscale * np.exp(-x / decay_time)


def fitexp(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[
        Tuple[Optional[float], Optional[float], Optional[float]]
    ] = None,
) -> Tuple[Tuple[float, float, float], np.ndarray]:
    """return (y0, yscale, decay_time), (pOpt, pCov)"""
    if fitparams is None:
        fitparams = (None, None, None)

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = np.mean(ydata[-5:]).item()
        yscale = ydata[0] - y0
        x_2 = xdata[np.argmin(np.abs(ydata - (y0 + yscale / 2)))]
        x_4 = xdata[np.argmin(np.abs(ydata - (y0 + yscale / 4)))]
        decay_time = (x_2 / np.log(2) + x_4 / np.log(4)) / 2

        assign_init_p(fitparams, [y0, yscale, decay_time])
    fitparams = cast(Tuple[float, float, float], tuple(fitparams))

    # bounds
    bounds = (
        [-np.inf, -2 * np.abs(fitparams[1]), 0],
        [np.inf, 2 * np.abs(fitparams[1]), np.inf],
    )

    return fit_func(xdata, ydata, expfunc, fitparams, bounds)  # type: ignore


def dual_expfunc(x: NDArray[np.float64], *p: float) -> NDArray[np.float64]:
    """p = [y0, yscale1, decay_time1, yscale2, decay_time2]"""
    y0, yscale1, decay_time1, yscale2, decay_time2 = p
    return y0 + yscale1 * np.exp(-x / decay_time1) + yscale2 * np.exp(-x / decay_time2)


def fit_dualexp(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[
        Tuple[
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
        ]
    ] = None,
) -> Tuple[Tuple[float, float, float, float, float], np.ndarray]:
    """return (y0, yscale1, decay_time1, yscale2, decay_time2), (pOpt, pCov)"""
    if fitparams is None:
        fitparams = (None, None, None, None, None)

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = np.mean(ydata[-5:]).item()

        mid_idx = len(ydata) // 2
        xdata1, xdata2 = xdata[:mid_idx], xdata[mid_idx:]
        ydata1, ydata2 = ydata[:mid_idx], ydata[mid_idx:]

        yscale1 = ydata1[0] - y0
        x_2 = xdata1[np.argmin(np.abs(ydata1 - (y0 + yscale1 / 2)))] - xdata1[0]
        x_4 = xdata1[np.argmin(np.abs(ydata1 - (y0 + yscale1 / 4)))] - xdata1[0]
        decay1 = (x_2 / np.log(2) + x_4 / np.log(4)) / 2

        yscale1 *= np.exp(xdata[mid_idx] / decay1)

        yscale2 = ydata2[0] - y0
        x_2 = xdata2[np.argmin(np.abs(ydata2 - (y0 + yscale2 / 2)))] - xdata2[0]
        x_4 = xdata2[np.argmin(np.abs(ydata2 - (y0 + yscale2 / 4)))] - xdata2[0]
        decay2 = (x_2 / np.log(2) + x_4 / np.log(4)) / 2

        yscale2 *= np.exp(xdata[mid_idx] / decay2)

        yscale1 /= 2
        yscale2 /= 2

        assign_init_p(fitparams, [y0, yscale1, decay1, yscale2, decay2])
    fitparams = cast(Tuple[float, float, float, float, float], tuple(fitparams))

    # bounds
    bounds = (
        [-np.inf, -2 * np.abs(fitparams[1]), 0, -2 * np.abs(fitparams[3]), 0],
        [np.inf, 2 * np.abs(fitparams[1]), np.inf, 2 * np.abs(fitparams[3]), np.inf],
    )

    return fit_func(xdata, ydata, dual_expfunc, fitparams, bounds)  # type: ignore
