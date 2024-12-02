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
