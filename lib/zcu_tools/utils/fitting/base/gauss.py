import numpy as np

from .base import batch_fit_func, fit_func


# Gaussian function
def gauss_func(x, yscale, x_c, sigma):
    return yscale * np.exp(-0.5 * ((x - x_c) / sigma) ** 2)


def fit_gauss(xdata, ydata, fixedparams=None):
    if fixedparams is not None and len(fixedparams) != 3:
        raise ValueError(
            "Fixed parameters must be a list of three elements: [yscale, x_c, sigma]"
        )

    # guess initial parameters
    max_idx = np.argmax(np.abs(ydata))
    yscale = ydata[max_idx]
    x_c = xdata[max_idx]
    sigma = (
        0.5
        * (xdata.max() - xdata.min())
        * np.sum(np.abs(ydata) > np.abs(yscale) / 2)
        / len(xdata)
    )
    fitparams = [yscale, x_c, sigma]

    # bounds
    bounds = (
        [-2 * np.abs(fitparams[0]), xdata.min(), 0],
        [2 * np.abs(fitparams[0]), xdata.max(), xdata.max() - xdata.min()],
    )

    return fit_func(xdata, ydata, gauss_func, fitparams, bounds, fixedparams)


# # Dual Gaussian function
def dual_gauss_func(x, yscale1, x_c1, sigma1, yscale2, x_c2, sigma2):
    return gauss_func(x, yscale1, x_c1, sigma1) + gauss_func(x, yscale2, x_c2, sigma2)


def guess_dual_gauss_params(xdata, ydata):
    abs_ydata = np.abs(ydata)

    # guess initial parameters
    max_idx = np.argmax(abs_ydata)
    yscale1 = abs_ydata[max_idx]
    x_c1 = xdata[max_idx]
    sigma1 = (
        0.25
        * (xdata.max() - xdata.min())
        * np.sum(abs_ydata > yscale1 / 2)
        / len(xdata)
    )

    mean_x = np.sum(abs_ydata * xdata) / np.sum(abs_ydata)

    x_c2 = 2 * mean_x - x_c1
    c2_idx = np.argmin(np.abs(xdata - x_c2))
    yscale2 = abs_ydata[c2_idx]
    sigma2 = sigma1

    if x_c1 < x_c2:  # make first peak left
        return [yscale1, x_c1, sigma1, yscale2, x_c2, sigma2]
    else:
        return [yscale2, x_c2, sigma2, yscale1, x_c1, sigma1]


def fit_dual_gauss(xdata, ydata, fixedparams=None):
    if fixedparams is not None and len(fixedparams) != 6:
        raise ValueError(
            "Fixed parameters must be a list of six elements: [yscale1, x_c1, sigma1, yscale2, x_c2, sigma2]"
        )

    fitparams = guess_dual_gauss_params(xdata, ydata)

    # bounds
    bounds = (
        [
            0.0,
            xdata.min(),
            0,
            0.0,
            fitparams[4],
            0,
        ],
        [
            2 * max(fitparams[0], fitparams[3]),
            fitparams[1],
            xdata.max() - xdata.min(),
            2 * max(fitparams[0], fitparams[3]),
            xdata.max(),
            xdata.max() - xdata.min(),
        ],
    )

    params, pcov = fit_func(
        xdata, ydata, dual_gauss_func, fitparams, bounds, fixedparams
    )

    return params, pcov


def fit_dual_gauss_gmm(signals):
    """
    Params:
    -----------
    signals:
        1d / 2d real array
    Return:
    -----------
    params:
        [yscale1, x_c1, sigma1, yscale2, x_c2, sigma2]
        note that x_c1 < x_c2 and is the first dim of gauss center
    """
    from sklearn.mixture import GaussianMixture

    if signals.dtype == complex:
        raise ValueError("signals shape most be real array")
    if len(signals.shape) == 1:
        signals = signals[:, None]

    gmm = GaussianMixture(n_components=2, covariance_type="spherical")
    gmm.fit(signals)

    yscale1 = signals.shape[0] * gmm.weights_[0]
    yscale2 = signals.shape[1] * gmm.weights_[1]
    x_c1, x_c2 = gmm.means_[0][0], gmm.means_[1][0]
    sigma1 = np.sqrt(gmm.covariances_[0])
    sigma2 = np.sqrt(gmm.covariances_[1])

    if x_c1 < x_c2:  # make first peak left
        return [yscale1, x_c1, sigma1, yscale2, x_c2, sigma2]
    else:
        return [yscale2, x_c2, sigma2, yscale1, x_c1, sigma1]


def batch_fit_dual_gauss(list_xdata, list_ydata, list_init_p0=None, fixedparams=None):
    if list_init_p0 is None:
        list_init_p0 = [None] * len(list_xdata)

    # guess initial parameters
    list_fitparams = []
    list_bounds = []
    for xdata, ydata, init_p0 in zip(list_xdata, list_ydata, list_init_p0):
        # guess initial parameters
        params = guess_dual_gauss_params(xdata, ydata)

        if init_p0 is not None:
            for i, p0 in enumerate(init_p0):
                if p0 is not None:
                    params[i] = p0

        list_fitparams.append(params)

        list_bounds.append(
            (
                [
                    0.0,
                    xdata.min(),
                    0,
                    0.0,
                    params[4],
                    0,
                ],
                [
                    2 * max(params[0], params[3]),
                    params[1],
                    xdata.max() - xdata.min(),
                    2 * max(params[0], params[3]),
                    xdata.max(),
                    xdata.max() - xdata.min(),
                ],
            )
        )

    # let x_c1, sigma1, x_c2, sigma2 be shared
    # and yscale1, yscale2 be local
    shared_idxs = [1, 2, 4, 5]

    list_popts, list_covs = batch_fit_func(
        list_xdata,
        list_ydata,
        dual_gauss_func,
        list_fitparams,
        shared_idxs,
        list_bounds=list_bounds,
        fixedparams=fixedparams,
    )
    list_popts = np.array(list_popts)
    list_covs = np.array(list_covs)

    return list_popts, list_covs
