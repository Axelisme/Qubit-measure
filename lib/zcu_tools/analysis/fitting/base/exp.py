from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .base import FitResult, assign_init_p, fit_func


# exponential decay function
def expfunc(x: NDArray[np.float64], *p: float) -> NDArray[np.float64]:
    """p = [y0, yscale, decay_time]"""
    y0, yscale, decay_time = p
    return y0 + yscale * np.exp(-x / decay_time)


def fitexp(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    fitparams: Sequence[float | None] | NDArray[np.float64] | None = None,
    fixedparams: Sequence[float | None] | None = None,
    *,
    decay_bounds: tuple[float, float] | None = None,
) -> FitResult:
    """Fit [y0, yscale, decay_time] and covariance to xdata/ydata.

    fitparams supplies initial guesses; non-None fixedparams hold values fixed.
    Optional decay_bounds=(lower, upper) constrains free decay time in x units
    and clips its initial guess. None keeps the unconstrained positive time.
    Numeric fitting failures raise AnalysisFailed; invalid bounds raise ValueError.
    """
    if fitparams is None:
        fitparams = [None] * 3
    fitparams = list(fitparams)

    if fixedparams is not None and len(fixedparams) != 3:
        raise ValueError(
            "Fixed parameters must be a list of three elements: [y0, yscale, decay_time]"
        )

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = np.mean(ydata[-5:]).item()
        yscale = ydata[0] - y0
        x_2 = xdata[np.argmin(np.abs(ydata - (y0 + yscale / 2)))]
        x_4 = xdata[np.argmin(np.abs(ydata - (y0 + yscale / 4)))]
        decay_time = (x_2 / np.log(2) + x_4 / np.log(4)) / 2

        assign_init_p(fitparams, [y0, yscale, decay_time])
    # tolerated(cast): Baseline/amplitude/decay guesses fill unset slots and preserve supplied values; mutation cannot narrow the list type.
    fitparams = cast(list[float], fitparams)

    lower, upper = (0.0, np.inf) if decay_bounds is None else decay_bounds
    if decay_bounds is not None:
        fitparams[2] = min(max(fitparams[2], lower), upper)
    bounds = (
        [-np.inf, -2 * np.abs(fitparams[1]), lower],
        [np.inf, 2 * np.abs(fitparams[1]), upper],
    )

    return fit_func(xdata, ydata, expfunc, fitparams, bounds, fixedparams=fixedparams)


def dual_expfunc(x: NDArray[np.float64], *p: float) -> NDArray[np.float64]:
    """p = [y0, yscale1, decay_time1, yscale2, decay_time2]"""
    y0, yscale1, decay_time1, yscale2, decay_time2 = p
    return y0 + yscale1 * np.exp(-x / decay_time1) + yscale2 * np.exp(-x / decay_time2)


def fit_dualexp(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    fitparams: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
    *,
    decay_bounds: tuple[float, float] | None = None,
) -> FitResult:
    """Fit [y0, scale1, time1, scale2, time2] and covariance to xdata/ydata.

    fitparams supplies initial guesses; non-None fixedparams hold values fixed.
    Optional decay_bounds=(lower, upper) constrains both free decay times in x
    units and clips their initial guesses. None keeps positive unbounded times.
    Numeric fitting failures raise AnalysisFailed; invalid bounds raise ValueError.
    """
    if fitparams is None:
        fitparams = [None] * 5
    fitparams = list(fitparams)

    if fixedparams is not None and len(fixedparams) != 5:
        raise ValueError(
            "Fixed parameters must be a list of five elements: [y0, yscale1, decay_time1, yscale2, decay_time2]"
        )

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
    # tolerated(cast): Baseline/two-decay guesses fill unset slots and preserve supplied values; mutation cannot narrow the list type.
    fitparams = cast(list[float], fitparams)

    lower, upper = (0.0, np.inf) if decay_bounds is None else decay_bounds
    if decay_bounds is not None:
        for index in (2, 4):
            fitparams[index] = min(max(fitparams[index], lower), upper)
    bounds = (
        [-np.inf, -2 * np.abs(fitparams[1]), lower, -2 * np.abs(fitparams[3]), lower],
        [np.inf, 2 * np.abs(fitparams[1]), upper, 2 * np.abs(fitparams[3]), upper],
    )

    return fit_func(
        xdata, ydata, dual_expfunc, fitparams, bounds, fixedparams=fixedparams
    )
