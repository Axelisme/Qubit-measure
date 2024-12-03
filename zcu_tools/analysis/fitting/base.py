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
