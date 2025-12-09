from typing import List, Optional, Sequence, cast

import numpy as np

from .base import assign_init_p, batch_fit_func, fit_func


# Gaussian function
def gauss_func(x, y0, yscale, x_c, sigma):
    """params: [y0, yscale, x_c, sigma]"""
    return y0 + yscale * np.exp(-0.5 * ((x - x_c) / sigma) ** 2)


def fit_gauss(
    xdata,
    ydata,
    fitparams: Optional[Sequence[Optional[float]]] = None,
    fixedparams: Optional[Sequence[Optional[float]]] = None,
):
    """params: [y0, yscale, x_c, sigma]"""
    if fixedparams is not None and len(fixedparams) != 4:
        raise ValueError(
            "Fixed parameters must be a list of four elements: [y0, yscale, x_c, sigma]"
        )

    if fitparams is None:
        fitparams = [None] * 4

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
    fitparams = cast(List[float], fitparams)

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


# # Dual Gaussian function
def dual_gauss_func(x, yscale1, x_c1, sigma1, yscale2, x_c2, sigma2):
    return gauss_func(x, 0.0, yscale1, x_c1, sigma1) + gauss_func(
        x, 0.0, yscale2, x_c2, sigma2
    )


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

    means = gmm.means_
    weights = gmm.weights_
    covariances = gmm.covariances_

    assert means is not None
    assert weights is not None
    assert covariances is not None

    yscale1 = signals.shape[0] * weights[0]
    yscale2 = signals.shape[1] * weights[1]
    x_c1, x_c2 = means[0][0], means[1][0]
    sigma1 = np.sqrt(covariances[0])
    sigma2 = np.sqrt(covariances[1])

    if x_c1 < x_c2:  # make first peak left
        return [yscale1, x_c1, sigma1, yscale2, x_c2, sigma2]
    else:
        return [yscale2, x_c2, sigma2, yscale1, x_c1, sigma1]
