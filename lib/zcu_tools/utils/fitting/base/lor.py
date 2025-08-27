import numpy as np

from .base import assign_init_p, fit_func


# lorentzian function
def lorfunc(x, *p):
    y0, slope, yscale, x0, gamma = p
    return y0 + slope * (x - x0) + yscale / (1 + ((x - x0) / gamma) ** 2)


def fitlor(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 5

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = (ydata[0] + ydata[-1]) / 2
        slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        curve_up = np.max(ydata) + np.min(ydata) < 2 * y0
        if curve_up:
            yscale = np.min(ydata) - y0
            x0 = xdata[np.argmin(ydata)]
        else:
            yscale = np.max(ydata) - y0
            x0 = xdata[np.argmax(ydata)]
        gamma = np.abs(yscale) / 10

        assign_init_p(fitparams, [y0, slope, yscale, x0, gamma])

    # bounds
    yscale = fitparams[2]
    bounds = (
        [np.min(ydata), -np.inf, -2 * np.abs(yscale), -np.inf, 0],
        [np.max(ydata), np.inf, 2 * np.abs(yscale), np.inf, np.inf],
    )

    return fit_func(xdata, ydata, lorfunc, fitparams, bounds)


# asymmtric lorentzian function
def asym_lorfunc(x, *p):
    y0, slope, yscale, x0, gamma, alpha = p
    return (
        y0
        + slope * (x - x0)
        + yscale / (1 + ((x - x0) / (gamma * (1 + alpha * (x - x0)))) ** 2)
    )


def fit_asym_lor(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 6

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = (ydata[0] + ydata[-1]) / 2
        slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        curve_up = np.max(ydata) + np.min(ydata) < 2 * y0
        if curve_up:
            yscale = np.min(ydata) - y0
            x0 = xdata[np.argmin(ydata)]
        else:
            yscale = np.max(ydata) - y0
            x0 = xdata[np.argmax(ydata)]
        # gamma = (xdata[-1] - xdata[0]) / 100
        # calculate gamma from variance
        # weights = np.abs(ydata - np.median(ydata))
        # weights = np.where(weights > 0.7 * np.max(weights), 0, weights)
        # weights = weights / np.sum(weights)
        # gamma = np.sqrt(np.sum(weights * (xdata - x0) ** 2)) / 10
        # calculate alpha from skewness
        # skewness = np.sum(weights * (xdata - x0) ** 3) / gamma**3
        # alpha = -skewness / 5
        gamma = np.abs(yscale) / 10
        alpha = 0

        assign_init_p(fitparams, [y0, slope, yscale, x0, gamma, alpha])

    # bounds
    yscale = fitparams[2]
    bounds = (
        [-np.inf, -np.inf, -2 * np.abs(yscale), -np.inf, 0, -np.inf],
        [np.inf, np.inf, 2 * np.abs(yscale), np.inf, np.inf, np.inf],
    )

    return fit_func(xdata, ydata, asym_lorfunc, fitparams, bounds)
