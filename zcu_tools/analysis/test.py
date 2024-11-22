import numpy as np
import unittest

from .fitting import (
    fitexp,
    fitlor,
    fit_asym_lor,
    fitsin,
    fitdecaysin,
    expfunc,
    lorfunc,
    asym_lorfunc,
    sinfunc,
    decaysin,
)
import random


class TestFitting(unittest.TestCase):
    times = 100

    def test_fitexp(self):
        xdata = np.linspace(0, 10, 100)

        print("Testing fitexp...", end="")

        errs = []
        for i in range(self.times):
            y0 = 10 * (random.random() - 0.5)
            yscale = 20 * (random.random() - 0.5)
            decay = 1.5 + 1 * (random.random() - 0.5)
            ydata = expfunc(xdata, y0, yscale, decay)
            ydata += 0.05 * yscale * np.random.randn(len(ydata))  # add noise

            pOpt, pCov = fitexp(xdata, ydata)

            errs.append(np.abs(pOpt[2] / decay - 1))

            if errs[-1] > 0.2:
                print("Error:", errs[-1])
                print("pOpt:", pOpt)
                print([y0, yscale, decay])

        print(f"average error: {np.mean(errs):.2%}")
        self.assertLess(np.mean(errs), 0.05)

    def test_fitlor(self):
        xdata = np.linspace(0, 10, 100)

        print("Testing fitlor...", end="")

        errs = []
        for i in range(self.times):
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
            ydata += 0.05 * yscale * np.random.randn(len(ydata))

            pOpt, pCov = fitlor(xdata, ydata)

            errs.append(np.abs(pOpt[3] / x0 - 1))

        print(f"average error: {np.mean(errs):.2%}")
        self.assertLess(np.mean(errs), 0.01)

    def test_fitasymlor(self):
        xdata = np.linspace(0, 10, 100)

        print("Testing fit_asym_lor...", end="")

        errs = []
        for i in range(self.times):
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
            ydata += 0.05 * yscale * np.random.randn(len(ydata))

            pOpt, pCov = fit_asym_lor(xdata, ydata)

            errs.append(np.abs(pOpt[3] / x0 - 1))

        print(f"average error: {np.mean(errs):.2%}")
        self.assertLess(np.mean(errs), 0.01)

    def test_fitsin(self):
        xdata = np.linspace(0, 10, 100)

        print("Testing fitsin...", end="")

        errs = []
        for i in range(self.times):
            y0 = 100 * (random.random() - 0.5)
            yscale = 5 * random.random() + 5
            freq = 0.05 + 1 * random.random()
            phase = 360 * random.random()
            ydata = sinfunc(xdata, y0, yscale, freq, phase)
            ydata += 0.05 * yscale * np.random.randn(len(ydata))

            pOpt, pCov = fitsin(xdata, ydata)

            errs.append(np.abs(pOpt[2] / freq - 1))

        print(f"average error: {np.mean(errs):.2%}")
        self.assertLess(np.mean(errs), 0.01)

    def test_fitdecaysin(self):
        xdata = np.linspace(0, 10, 100)

        print("Testing fitdecaysin...", end="")

        errs = []
        for i in range(self.times):
            y0 = 100 * (random.random() - 0.5)
            yscale = 5 * random.random() + 5
            freq = 0.1 + 1 * random.random()
            phase = 360 * random.random()
            decay = 10 + 5 * random.random()
            ydata = decaysin(xdata, y0, yscale, freq, phase, decay)
            ydata += 0.05 * yscale * np.random.randn(len(ydata))

            pOpt, pCov = fitdecaysin(xdata, ydata)

            errs.append(np.abs(pOpt[2] / freq - 1))

        print(f"average error: {np.mean(errs):.2%}")
        self.assertLess(np.mean(errs), 0.01)


if __name__ == "__main__":
    unittest.main()
