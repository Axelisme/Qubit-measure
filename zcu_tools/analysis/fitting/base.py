import numpy as np
import scipy as sp


def fit_func(xdata, ydata, fitfunc, init_p=None, bounds=None, **kwargs):
    try:
        pOpt, pCov = sp.optimize.curve_fit(
            fitfunc, xdata, ydata, p0=init_p, bounds=bounds, **kwargs
        )
    except RuntimeError as e:
        print("Warning: fit failed!")
        print(e)
        pOpt = init_p
        pCov = np.full(shape=(len(init_p), len(init_p)), fill_value=np.inf)
    return pOpt, pCov


def assign_init_p(fitparams, init_p):
    for i, p in enumerate(init_p):
        if fitparams[i] is None:
            fitparams[i] = p
    return fitparams


def fit_line(xdata, ydata, fitparams=None):
    def fitfunc(x, a, b):
        return a * x + b

    if fitparams is None:
        fitparams = [None] * 2

    if any([p is None for p in fitparams]):
        a = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        b = ydata[0] - a * xdata[0]
        assign_init_p(fitparams, [a, b])

    bounds = (
        [-np.inf, -np.inf],
        [np.inf, np.inf],
    )

    return fit_func(xdata, ydata, fitfunc, fitparams, bounds)
