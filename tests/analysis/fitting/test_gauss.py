import unittest

import numpy as np

from lib.zcu_tools.analysis.fitting.gauss import (
    dual_gauss_func,
    fit_dual_gauss,
    fit_gauss,
    gauss_func,
)


class TestGauss(unittest.TestCase):
    def test_gauss_func(self):
        x = np.linspace(-5, 5, 100)
        params = [1, 0, 1]  # yscale, x_c, sigma
        y = gauss_func(x, *params)
        self.assertEqual(len(y), len(x))
        self.assertAlmostEqual(max(y), params[0], delta=0.05)

    def test_fit_gauss(self):
        xdata = np.linspace(-5, 5, 100)
        params = [2, 0.5, 1.5]
        ydata = gauss_func(xdata, *params) + np.random.normal(0, 0.1, len(xdata))

        fitparams, pcov = fit_gauss(xdata, ydata)
        self.assertTrue(np.allclose(fitparams, params, atol=0.2))

    def test_dual_gauss_func(self):
        x = np.linspace(-5, 5, 100)
        params = [1, -1, 1, 0.5, 2, 0.8]  # yscale1, x_c1, sigma1, yscale2, x_c2, sigma2
        y = dual_gauss_func(x, *params)
        self.assertEqual(len(y), len(x))

    def test_fit_dual_gauss(self):
        xdata = np.linspace(-5, 5, 100)
        params = [1, -1, 1, 0.5, 2, 0.8]
        ydata = dual_gauss_func(xdata, *params) + np.random.normal(0, 0.1, len(xdata))

        fitparams, pcov = fit_dual_gauss(xdata, ydata)
        self.assertTrue(np.allclose(fitparams, params, atol=0.3))


if __name__ == "__main__":
    unittest.main()
