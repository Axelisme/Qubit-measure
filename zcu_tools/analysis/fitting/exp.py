import numpy as np

from .base import assign_init_p, fit_func


# exponential decay function
def expfunc(x, *p):
    y0, yscale, decay = p
    return y0 + yscale * np.exp(-x / decay)


def fitexp(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = np.mean(ydata[-5:])
        yscale = ydata[0] - y0
        x_2 = xdata[np.argmin(np.abs(ydata - (y0 + yscale / 2)))]
        x_4 = xdata[np.argmin(np.abs(ydata - (y0 + yscale / 4)))]
        decay = (x_2 / np.log(2) + x_4 / np.log(4)) / 2

        assign_init_p(fitparams, [y0, yscale, decay])

    # bounds
    bounds = (
        [-np.inf, -2 * np.abs(fitparams[1]), 0],
        [np.inf, 2 * np.abs(fitparams[1]), np.inf],
    )

    return fit_func(xdata, ydata, expfunc, fitparams, bounds)


def dual_expfunc(x, *p):
    y0, yscale1, decay1, yscale2, decay2 = p
    return y0 + yscale1 * np.exp(-x / decay1) + yscale2 * np.exp(-x / decay2)


def fit_dualexp(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 5

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = np.mean(ydata[-5:])

        mid_idx = len(ydata) // 2
        ydata1, ydata2 = ydata[:mid_idx], ydata[mid_idx:]

        yscale1 = ydata1[0] - y0
        x_2 = xdata[np.argmin(np.abs(ydata1 - (y0 + yscale1 / 2)))]
        x_4 = xdata[np.argmin(np.abs(ydata1 - (y0 + yscale1 / 4)))]
        decay1 = (x_2 / np.log(2) + x_4 / np.log(4)) / 2

        yscale2 = ydata2[0] - y0
        x_2 = xdata[np.argmin(np.abs(ydata2 - (y0 + yscale2 / 2)))] - xdata[mid_idx]
        x_4 = xdata[np.argmin(np.abs(ydata2 - (y0 + yscale2 / 4)))] - xdata[mid_idx]
        decay2 = (x_2 / np.log(2) + x_4 / np.log(4)) / 2

        yscale2 *= np.exp(xdata[mid_idx] / decay2)

        assign_init_p(fitparams, [y0, yscale1, decay1, yscale2, decay2])

    # bounds
    bounds = (
        [-np.inf, -2 * np.abs(fitparams[1]), 0, -2 * np.abs(fitparams[3]), 0],
        [np.inf, 2 * np.abs(fitparams[1]), np.inf, 2 * np.abs(fitparams[3]), np.inf],
    )

    return fit_func(xdata, ydata, dual_expfunc, fitparams, bounds)
