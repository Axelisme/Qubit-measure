import numpy as np

from .base import assign_init_p, fit_func


# sinc函數模型
def sincfunc(x, *p):
    y0, slope, yscale, x0, gamma = p
    return y0 + slope * (x - x0) + yscale * np.sinc((x - x0) / gamma)


# 擬合sinc函數
def fitsinc(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 5

    # 初始參數猜測
    if any(p is None for p in fitparams):
        y0 = (ydata[0] + ydata[-1]) / 2
        slope = (ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
        curve_up = np.max(ydata) + np.min(ydata) < 2 * y0
        if curve_up:
            yscale = np.min(ydata) - y0
            x0 = xdata[np.argmin(ydata)]
        else:
            yscale = np.max(ydata) - y0
            x0 = xdata[np.argmax(ydata)]
        gamma = (xdata[-1] - xdata[0]) / 10

        assign_init_p(fitparams, [y0, slope, yscale, x0, gamma])

    # 參數邊界
    yscale = fitparams[2]
    bounds = (
        [np.min(ydata), -np.inf, -2 * np.abs(yscale), -np.inf, 0],
        [np.max(ydata), np.inf, 2 * np.abs(yscale), np.inf, np.inf],
    )

    return fit_func(xdata, ydata, sincfunc, fitparams, bounds)
