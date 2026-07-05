from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .base import assign_init_p, fit_func


# lorentzian function
def lorfunc(x: NDArray[np.float64], *p: float) -> NDArray[np.float64]:
    """p = [y0, slope, yscale, x0, gamma]"""
    y0, slope, yscale, x0, gamma = p
    return y0 + slope * (x - x0) + yscale / (1 + ((x - x0) / gamma) ** 2)


def _guess_lorentzian_params(
    xdata: NDArray[np.float64], ydata: NDArray[np.float64]
) -> tuple[float, float, float, float, float]:
    y0 = float(np.median(ydata))
    slope = float((ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0]))
    max_prominence = float(np.max(ydata) - y0)
    min_prominence = float(y0 - np.min(ydata))
    # Use the median baseline so an edge peak/dip does not poison the endpoint
    # average and invert the initial peak-vs-dip decision.
    if max_prominence >= min_prominence:
        yscale = max_prominence
        x0 = float(xdata[np.argmax(ydata)])
    else:
        yscale = -min_prominence
        x0 = float(xdata[np.argmin(ydata)])
    gamma = np.abs(yscale) / 10
    return y0, slope, yscale, x0, gamma


def fitlor(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    fitparams: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[list[float], NDArray[np.float64]]:
    if fitparams is None:
        fitparams = [None] * 5
    fitparams = list(fitparams)

    # guess initial parameters
    if any([p is None for p in fitparams]):
        assign_init_p(fitparams, _guess_lorentzian_params(xdata, ydata))
    fitparams = cast(list[float], fitparams)

    # bounds
    yscale = fitparams[2]
    max_slope = (np.max(ydata) - np.min(ydata)) / (xdata.max() - xdata.min())
    bounds = (
        [np.min(ydata), -max_slope, -2 * np.abs(yscale), xdata.min(), 0],
        [np.max(ydata), max_slope, 2 * np.abs(yscale), xdata.max(), np.inf],
    )

    return fit_func(xdata, ydata, lorfunc, fitparams, bounds, fixedparams=fixedparams)


# asymmtric lorentzian function
def asym_lorfunc(x: NDArray[np.float64], *p: float) -> NDArray[np.float64]:
    """p = [y0, slope, yscale, x0, gamma, alpha]"""
    y0, slope, yscale, x0, gamma, alpha = p
    return (
        y0
        + slope * (x - x0)
        + yscale / (1 + ((x - x0) / (gamma * (1 + alpha * (x - x0)))) ** 2)
    )


def fit_asym_lor(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    fitparams: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[list[float], NDArray[np.float64]]:
    if fitparams is None:
        fitparams = [None] * 6
    fitparams = list(fitparams)

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0, slope, yscale, x0, gamma = _guess_lorentzian_params(xdata, ydata)
        alpha = 0

        assign_init_p(fitparams, [y0, slope, yscale, x0, gamma, alpha])
    fitparams = cast(list[float], fitparams)

    # bounds
    yscale = fitparams[2]
    bounds = (
        [-np.inf, -np.inf, -2 * np.abs(yscale), -np.inf, 0, -np.inf],
        [np.inf, np.inf, 2 * np.abs(yscale), np.inf, np.inf, np.inf],
    )

    return fit_func(
        xdata, ydata, asym_lorfunc, fitparams, bounds, fixedparams=fixedparams
    )
