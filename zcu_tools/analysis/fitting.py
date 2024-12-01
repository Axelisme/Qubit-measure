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


# ====================================================== #
"""
exponential decay function
"""


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


# ====================================================== #
"""
lorentzian function
"""


def lorfunc(x, *p):
    y0, slope, yscale, x0, gamma = p
    return y0 + slope * x + yscale / (1 + ((x - x0) / gamma) ** 2)


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


# ====================================================== #
"""
asymmtric lorentzian function
"""


def asym_lorfunc(x, *p):
    y0, slope, yscale, x0, gamma, alpha = p
    return (
        y0
        + slope * x
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
        gamma = (xdata[-1] - xdata[0]) / 10
        alpha = 0

        assign_init_p(fitparams, [y0, slope, yscale, x0, gamma, alpha])

    # bounds
    yscale = fitparams[2]
    bounds = (
        [np.min(ydata), -np.inf, -2 * np.abs(yscale), -np.inf, 0, -np.inf],
        [np.max(ydata), np.inf, 2 * np.abs(yscale), np.inf, np.inf, np.inf],
    )

    return fit_func(xdata, ydata, asym_lorfunc, fitparams, bounds)


# ====================================================== #
"""
sinusoidal function
"""


def sinfunc(x, *p):
    y0, yscale, freq, phase = p
    return y0 + yscale * np.sin(2 * np.pi * (freq * x + phase / 360))


def fitsin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 4

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = (np.max(ydata) + np.min(ydata)) / 2
        yscale = (np.max(ydata) - np.min(ydata)) / 2
        min_freq = 0.125 / (xdata[-1] - xdata[0])
        fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
        freq_mask = fft_freqs >= min_freq
        fft = np.fft.fft(ydata)[freq_mask]
        fft_freqs = fft_freqs[freq_mask]
        max_id = np.argmax(np.abs(fft))
        freq = fft_freqs[max_id]
        phase = np.angle(fft[max_id], deg=True) % 360

        assign_init_p(fitparams, [y0, yscale, freq, phase])

    # bounds
    yscale = fitparams[1]
    freq = fitparams[2]
    bounds = (
        [-np.inf, -2 * np.abs(yscale), 0.2 * freq, -720],
        [np.inf, 2 * np.abs(yscale), 5 * freq, 720],
    )

    pOpt, pCov = fit_func(xdata, ydata, sinfunc, fitparams, bounds)
    if pOpt[1] < 0:
        pOpt[1] = -pOpt[1]
        pOpt[3] = pOpt[3] + 180
    pOpt[3] = pOpt[3] % 360  # convert phase to 0-360
    return pOpt, pCov


# ====================================================== #
"""
damped sinusoidal function
"""


def decaysin(x, *p):
    y0, yscale, freq, phase, decay = p
    return y0 + yscale * np.sin(2 * np.pi * (freq * x + phase / 360)) * np.exp(
        -x / decay
    )


def fitdecaysin(xdata, ydata, fitparams=None):
    if fitparams is None:
        fitparams = [None] * 5

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = (np.max(ydata) + np.min(ydata)) / 2
        yscale = (np.max(ydata) - np.min(ydata)) / 2
        min_freq = 0.125 / (xdata[-1] - xdata[0])
        fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
        freq_mask = fft_freqs >= min_freq
        fft = np.fft.fft(ydata)[freq_mask]
        fft_freqs = fft_freqs[freq_mask]
        max_id = np.argmax(np.abs(fft))
        freq = fft_freqs[max_id]
        phase = np.angle(fft[max_id], deg=True) % 360
        decay = xdata[-1] - xdata[0]

        assign_init_p(fitparams, [y0, yscale, freq, phase, decay])

    # bounds
    yscale = fitparams[1]
    freq = fitparams[2]
    decay = fitparams[4]
    bounds = (
        [-np.inf, -2 * np.abs(yscale), 0.2 * freq, -720, 0],
        [np.inf, 2 * np.abs(yscale), 5 * freq, 720, np.inf],
    )

    pOpt, pCov = fit_func(xdata, ydata, decaysin, fitparams, bounds)
    if pOpt[1] < 0:
        pOpt[1] = -pOpt[1]
        pOpt[3] = pOpt[3] + 180
    pOpt[3] = pOpt[3] % 360  # convert phase to 0-360
    return pOpt, pCov
