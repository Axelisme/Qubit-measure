from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .base import FitResult, assign_init_p, fit_func


# Gaussian function
def gauss_func(
    x: NDArray[np.float64], y0: float, yscale: float, x_c: float, sigma: float
) -> NDArray[np.float64]:
    """params: [y0, yscale, x_c, sigma]"""
    return y0 + yscale * np.exp(-0.5 * ((x - x_c) / sigma) ** 2)


def fit_gauss(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    fitparams: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> FitResult:
    """params: [y0, yscale, x_c, sigma]"""
    if fixedparams is not None and len(fixedparams) != 4:
        raise ValueError(
            "Fixed parameters must be a list of four elements: [y0, yscale, x_c, sigma]"
        )

    if fitparams is None:
        fitparams = [None] * 4
    fitparams = list(fitparams)

    # guess initial parameters
    if any([p is None for p in fitparams]):
        # guess initial parameters
        if np.max(ydata) + np.min(ydata) > 2 * np.mean(ydata):
            y0 = np.min(ydata)
        else:
            y0 = np.max(ydata)
        norm_ydata = np.abs(ydata - y0)
        yscale = np.max(ydata)
        max_idx = np.argmax(norm_ydata)
        x_c = xdata[max_idx]
        sigma = (
            0.5
            * (xdata.max() - xdata.min())
            / (len(xdata) - 1)
            * np.sum(norm_ydata > 0.5 * np.abs(yscale))
        )
        assign_init_p(fitparams, [y0, yscale, x_c, sigma])
    # tolerated(cast): reportArgumentType: Baseline/amplitude/center/width guesses fill unset slots and preserve supplied values; mutation cannot narrow the list type.
    fitparams = cast(list[float], fitparams)

    # bounds
    y0, yscale, x_c, sigma = fitparams
    bounds = (
        [
            y0 - 0.5 * np.abs(yscale),
            -2 * np.abs(yscale),
            xdata.min(),
            0,
        ],
        [
            y0 + 0.5 * np.abs(yscale),
            2 * np.abs(yscale),
            xdata.max(),
            xdata.max() - xdata.min(),
        ],
    )

    return fit_func(xdata, ydata, gauss_func, fitparams, bounds, fixedparams)
