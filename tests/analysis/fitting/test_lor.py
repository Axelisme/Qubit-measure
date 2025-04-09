import unittest

import numpy as np

from lib.zcu_tools.analysis.fitting.lor import (
    lorfunc,
    fitlor,
    asym_lorfunc,
    fit_asym_lor,
)


class TestLor(unittest.TestCase):
    def test_lorfunc(self):
        x = np.linspace(-10, 10, 200)
        params = [1, 0.1, 3, 0, 2]  # y0, slope, yscale, x0, gamma
        y = lorfunc(x, *params)
        self.assertEqual(len(y), len(x))

        # Verify peak position
        peak_index = np.argmax(y)
        self.assertAlmostEqual(x[peak_index], params[3], delta=0.1)

        # Verify peak height
        self.assertAlmostEqual(y[peak_index], params[0] + params[2], delta=0.1)

        # Verify FWHM (Full Width at Half Maximum)
        half_max = params[0] + params[2] / 2
        indices = np.where(y > half_max)[0]
        width = x[indices[-1]] - x[indices[0]]
        expected_fwhm = 2 * params[4]  # FWHM = 2*gamma for Lorentzian
        self.assertAlmostEqual(width, expected_fwhm, delta=0.5)

    def test_fitlor(self):
        xdata = np.linspace(-10, 10, 200)
        params = [1, 0.1, 3, 0, 2]  # y0, slope, yscale, x0, gamma
        ydata = lorfunc(xdata, *params) + np.random.normal(0, 0.1, len(xdata))

        fitparams, _ = fitlor(xdata, ydata)

        # Check if the fitted parameters are close to the original ones
        self.assertTrue(np.allclose(fitparams, params, atol=0.3))

    def test_asym_lorfunc(self):
        x = np.linspace(-10, 10, 200)
        params = [1, 0.1, 3, 0, 2, 0.2]  # y0, slope, yscale, x0, gamma, alpha
        y = asym_lorfunc(x, *params)
        self.assertEqual(len(y), len(x))

        # Check asymmetry by comparing distances between peak and half-max points
        peak_index = np.argmax(y)
        peak_x = x[peak_index]
        half_max = params[0] + params[2] / 2

        # Find left and right points where y crosses half_max
        left_indices = np.where((y[:peak_index] < half_max))[0]
        right_indices = np.where((y[peak_index:] < half_max))[0]

        if len(left_indices) > 0 and len(right_indices) > 0:
            left_x = x[left_indices[-1]]
            right_x = x[peak_index + right_indices[0]]

            # For positive alpha, the right side should be broader
            if params[5] > 0:
                self.assertGreater(right_x - peak_x, peak_x - left_x)
            elif params[5] < 0:
                self.assertLess(right_x - peak_x, peak_x - left_x)

    def test_fit_asym_lor(self):
        xdata = np.linspace(-10, 10, 200)
        params = [1, 0.1, 3, 0, 2, 0.2]  # y0, slope, yscale, x0, gamma, alpha
        ydata = asym_lorfunc(xdata, *params) + np.random.normal(0, 0.1, len(xdata))

        fitparams, _ = fit_asym_lor(xdata, ydata)

        # Calculate fitted curve
        y_fit = asym_lorfunc(xdata, *fitparams)
        rmse = np.sqrt(np.mean((y_fit - ydata) ** 2))

        # The RMSE should be close to the noise level we added
        self.assertLess(rmse, 0.15)

        # Check key parameters
        self.assertAlmostEqual(fitparams[0], params[0], delta=0.3)  # y0
        self.assertAlmostEqual(fitparams[3], params[3], delta=0.3)  # x0

        # The sign of alpha should be consistent
        if abs(fitparams[5]) > 0.05:  # Only check if alpha is significantly non-zero
            self.assertEqual(np.sign(fitparams[5]), np.sign(params[5]))


if __name__ == "__main__":
    unittest.main()
