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


def test_fitexp(times=100):
    import random

    xdata = np.linspace(0, 10, 100)

    print("Testing fitexp...")

    errs = []
    for i in range(times):
        y0 = 10 * (random.random() - 0.5)
        yscale = 20 * (random.random() - 0.5)
        decay = 1.5 + 1 * (random.random() - 0.5)
        ydata = y0 + yscale * np.exp(-xdata / decay)
        ydata += 0.05 * yscale * np.random.randn(len(ydata))  # add noise

        pOpt, pCov = fitexp(xdata, ydata)

        errs.append(np.abs(pOpt[2] / decay - 1))

        if errs[-1] > 0.2:
            import matplotlib.pyplot as plt

            print("Error:", errs[-1])
            print("pOpt:", pOpt)
            print([y0, yscale, decay])

            plt.plot(xdata, ydata)
            plt.plot(xdata, expfunc(xdata, *pOpt))
            plt.legend(["data", "fit"])
            plt.show()

    print("\tAverage error:", np.mean(errs))


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
        [np.min(ydata), -np.inf, -2 * np.abs(yscale), xdata[0], 0],
        [np.max(ydata), np.inf, 2 * np.abs(yscale), xdata[-1], np.inf],
    )

    return fit_func(xdata, ydata, lorfunc, fitparams, bounds)


def test_fitlor(times=100):
    import random

    xdata = np.linspace(0, 10, 100)

    print("Testing fitlor...")

    errs = []
    for i in range(times):
        y0 = 100 * (random.random() - 0.5)
        slope = 1 * (random.random() - 0.5) / (xdata[-1] - xdata[0])
        gamma = 0.1 * (xdata[-1] - xdata[0]) * (random.random() + 0.2)
        yscale = 75 * gamma * (random.random() + 1)
        if random.random() > 0.5:
            yscale = -yscale
        x0 = (
            0.1 * (random.random() - 0.5) * (xdata[-1] - xdata[0])
            + (xdata[0] + xdata[-1]) / 2
        )
        ydata = lorfunc(xdata, y0, slope, yscale, x0, gamma)
        ydata += 0.05 * yscale * np.random.randn(len(ydata))  # add noise

        pOpt, pCov = fitlor(xdata, ydata)

        errs.append(np.abs(pOpt[3] / x0 - 1))

        if errs[-1] > 0.1:
            import matplotlib.pyplot as plt

            print("Error:", errs[-1])
            print("pOpt:", pOpt)
            print([y0, slope, yscale, x0, gamma])

            plt.plot(xdata, ydata)
            plt.plot(xdata, lorfunc(xdata, *pOpt))
            plt.legend(["data", "fit"])
            plt.show()

    print("\tAverage error:", np.mean(errs))


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
        [np.min(ydata), -np.inf, -2 * np.abs(yscale), xdata[0], 0, -np.inf],
        [np.max(ydata), np.inf, 2 * np.abs(yscale), xdata[-1], np.inf, np.inf],
    )

    return fit_func(xdata, ydata, asym_lorfunc, fitparams, bounds)


def test_fitasymlor(times=10):
    import random

    xdata = np.linspace(0, 10, 100)

    print("Testing fit_asym_lor...")

    errs = []
    for i in range(times):
        y0 = 100 * (random.random() - 0.5)
        slope = 1 * (random.random() - 0.5) / (xdata[-1] - xdata[0])
        gamma = 0.1 * (xdata[-1] - xdata[0]) * (random.random() + 0.2)
        yscale = 75 * gamma * (random.random() + 1)
        if random.random() > 0.5:
            yscale = -yscale
        x0 = (
            0.1 * (random.random() - 0.5) * (xdata[-1] - xdata[0])
            + (xdata[0] + xdata[-1]) / 2
        )
        alpha = 1 * (random.random() - 0.5)
        ydata = asym_lorfunc(xdata, y0, slope, yscale, x0, gamma, alpha)
        ydata += 0.05 * yscale * np.random.randn(len(ydata))  # add noise

        pOpt, pCov = fit_asym_lor(xdata, ydata)

        errs.append(np.abs(pOpt[3] / x0 - 1))

        if errs[-1] > 0.1:
            import matplotlib.pyplot as plt

            print("Error:", errs[-1])
            print("pOpt:", [round(p, 2) for p in pOpt])
            print([round(p, 2) for p in [y0, slope, yscale, x0, gamma, alpha]])

            plt.plot(xdata, ydata)
            plt.plot(xdata, asym_lorfunc(xdata, *pOpt))
            plt.legend(["data", "fit"])
            plt.show()

    print("\tAverage error:", np.mean(errs))


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


def test_fitsin(times=10):
    import random

    xdata = np.linspace(0, 10, 100)

    print("Testing fitsin...")

    errs = []
    for i in range(times):
        y0 = 100 * (random.random() - 0.5)
        yscale = 5 * random.random() + 5
        freq = 0.05 + 1 * random.random()
        phase = 360 * random.random()
        ydata = sinfunc(xdata, y0, yscale, freq, phase)
        ydata += 0.05 * yscale * np.random.randn(len(ydata))  # add noise

        pOpt, pCov = fitsin(xdata, ydata)

        errs.append(np.abs(pOpt[2] / freq - 1))

        if errs[-1] > 0.1:
            import matplotlib.pyplot as plt

            print("Error:", errs[-1])
            print("pOpt:", [round(p, 2) for p in pOpt])
            print([round(p, 2) for p in [y0, yscale, freq, phase]])

            plt.plot(xdata, ydata)
            plt.plot(xdata, sinfunc(xdata, *pOpt))
            plt.legend(["data", "fit"])
            plt.show()

    print("\tAverage error:", np.mean(errs))


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


def test_fitdecaysin(times=100):
    import random

    xdata = np.linspace(0, 10, 100)

    print("Testing fitdecaysin...")

    errs = []
    for i in range(times):
        y0 = 100 * (random.random() - 0.5)
        yscale = 5 * random.random() + 5
        freq = 0.1 + 1 * random.random()
        phase = 360 * random.random()
        decay = 10 + 5 * random.random()
        ydata = decaysin(xdata, y0, yscale, freq, phase, decay)
        ydata += 0.05 * yscale * np.random.randn(len(ydata))  # add noise

        pOpt, pCov = fitdecaysin(xdata, ydata)

        errs.append(np.abs(pOpt[2] / freq - 1))

        if errs[-1] > 0.1:
            import matplotlib.pyplot as plt

            print("Error:", errs[-1])
            print("pOpt:", [round(p, 2) for p in pOpt])
            print([round(p, 2) for p in [y0, yscale, freq, phase, decay]])

            plt.plot(xdata, ydata)
            plt.plot(xdata, decaysin(xdata, *pOpt))
            plt.legend(["data", "fit"])
            plt.show()

    print("\tAverage error:", np.mean(errs))


if __name__ == "__main__":
    test_fitexp()
    test_fitlor()
    test_fitasymlor()
    test_fitsin()
    test_fitdecaysin()
