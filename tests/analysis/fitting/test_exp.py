import unittest

import numpy as np

from lib.zcu_tools.analysis.fitting.exp import (
    dual_expfunc,
    expfunc,
    fit_dualexp,
    fitexp,
)


class TestExp(unittest.TestCase):
    def test_expfunc(self):
        x = np.linspace(0, 10, 100)
        params = [1, 2, 3]  # y0, yscale, decay
        y = expfunc(x, *params)
        self.assertEqual(len(y), len(x))

        # Verify the function value at x=0
        self.assertAlmostEqual(y[0], params[0] + params[1], delta=0.01)

        # Verify the decay by checking value at one decay time constant
        decay_idx = np.argmin(np.abs(x - params[2]))
        expected_value = params[0] + params[1] / np.e
        self.assertAlmostEqual(y[decay_idx], expected_value, delta=0.1)

    def test_fitexp(self):
        xdata = np.linspace(0, 10, 1000)
        params = [1, 2, 3]  # y0, yscale, decay
        ydata = expfunc(xdata, *params) + np.random.normal(0, 0.1, len(xdata))

        fitparams, _ = fitexp(xdata, ydata)

        # Check if the fitted parameters are close to the original ones
        self.assertTrue(np.allclose(fitparams, params, atol=0.4))

    def test_dual_expfunc(self):
        x = np.linspace(0, 20, 500)
        params = [1, 2, 1.5, 1, 5]  # y0, yscale1, decay1, yscale2, decay2
        y = dual_expfunc(x, *params)
        self.assertEqual(len(y), len(x))

        # Verify the function value at x=0
        expected_value_at_zero = params[0] + params[1] + params[3]
        self.assertAlmostEqual(y[0], expected_value_at_zero, delta=0.01)

        # Verify the final value approaches y0 for large x
        self.assertAlmostEqual(y[-1], params[0], delta=0.1)

    def test_fit_dualexp(self):
        xdata = np.linspace(0, 15, 200)
        params = [1, 2, 1.5, 1, 5]  # y0, yscale1, decay1, yscale2, decay2
        ydata = dual_expfunc(xdata, *params) + np.random.normal(0, 0.05, len(xdata))

        fitparams, _ = fit_dualexp(xdata, ydata)

        # This test is a bit more challenging due to potential parameter swapping
        # So we test the overall function fit rather than individual parameters
        y_fit = dual_expfunc(xdata, *fitparams)
        rmse = np.sqrt(np.mean((y_fit - ydata) ** 2))

        # The RMSE should be close to the noise level we added
        self.assertLess(rmse, 0.1)

        # Test that y0 is close to the original
        self.assertAlmostEqual(fitparams[0], params[0], delta=0.3)


if __name__ == "__main__":
    unittest.main()
