import numpy as np

from .base import assign_init_p, fit_func


# Gaussian function
def gauss_func(x, *p):
    yscale, x_c, sigma = p
    return yscale * np.exp(-0.5 * ((x - x_c) / sigma) ** 2)


def fit_gauss(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 3

    # guess initial parameters
    if any([p is None for p in fitparams]):
        max_idx = np.argmax(np.abs(ydata))
        yscale = ydata[max_idx]
        x_c = xdata[max_idx]
        sigma = (
            0.5
            * (xdata.max() - xdata.min())
            * np.sum(np.abs(ydata) > np.abs(yscale) / 2)
            / len(xdata)
        )

        assign_init_p(fitparams, [yscale, x_c, sigma])

    # bounds
    bounds = (
        [-2 * np.abs(fitparams[0]), xdata.min(), 0],
        [2 * np.abs(fitparams[0]), xdata.max(), xdata.max() - xdata.min()],
    )

    return fit_func(xdata, ydata, gauss_func, fitparams, bounds)


# # Dual Gaussian function
def dual_gauss_func(x, *p):
    return gauss_func(x, *p[:3]) + gauss_func(x, *p[3:])


def fit_dual_gauss(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 6

    # guess initial parameters
    if any([p is None for p in fitparams]):
        max_idx = np.argmax(np.abs(ydata))
        yscale1 = ydata[max_idx]
        x_c1 = xdata[max_idx]
        sigma1 = (
            0.25
            * (xdata.max() - xdata.min())
            * np.sum(np.abs(ydata) > np.abs(yscale1) / 2)
            / len(xdata)
        )

        mean_x = np.sum(ydata * xdata) / np.sum(ydata)

        x_c2 = 2 * mean_x - x_c1
        c2_idx = np.argmin(np.abs(xdata - x_c2))
        yscale2 = ydata[c2_idx]
        sigma2 = sigma1

        assign_init_p(fitparams, [yscale1, x_c1, sigma1, yscale2, x_c2, sigma2])

    # bounds
    bounds = (
        [
            -2 * np.abs(fitparams[0]),
            xdata.min(),
            0,
            -2 * np.abs(fitparams[3]),
            xdata.min(),
            0,
        ],
        [
            2 * np.abs(fitparams[0]),
            xdata.max(),
            xdata.max() - xdata.min(),
            2 * np.abs(fitparams[3]),
            xdata.max(),
            xdata.max() - xdata.min(),
        ],
    )

    return fit_func(xdata, ydata, dual_gauss_func, fitparams, bounds)
