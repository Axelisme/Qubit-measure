import unittest

import numpy as np

from lib.zcu_tools.analysis.fitting.cos import (
    cosfunc,
    fitcos,
    decaycos,
    fitdecaycos,
)


class TestCos(unittest.TestCase):
    def test_cosfunc(self):
        x = np.linspace(0, 10, 100)
        params = [1, 2, 0.5, 45]  # y0, yscale, freq, phase
        y = cosfunc(x, *params)
        self.assertEqual(len(y), len(x))

        # Check if the function generates a wave with the right amplitude
        self.assertAlmostEqual(np.max(y), params[0] + params[1], delta=0.1)
        self.assertAlmostEqual(np.min(y), params[0] - params[1], delta=0.1)

    def test_fitcos(self):
        xdata = np.linspace(0, 10, 100)
        params = [1, 2, 0.5, 45]  # y0, yscale, freq, phase
        ydata = cosfunc(xdata, *params) + np.random.normal(0, 0.1, len(xdata))

        fitparams, _ = fitcos(xdata, ydata)

        # Check if the fitted parameters are close to the original ones
        # Note: phase is modulo 360, so we need special handling
        self.assertAlmostEqual(fitparams[0], params[0], delta=0.2)  # y0
        self.assertAlmostEqual(fitparams[1], params[1], delta=0.2)  # yscale
        self.assertAlmostEqual(fitparams[2], params[2], delta=0.1)  # freq

        # For phase, we need to handle the wrapping around 360 degrees
        phase_diff = abs((fitparams[3] - params[3]) % 360)
        phase_diff = min(phase_diff, 360 - phase_diff)
        self.assertLess(phase_diff, 10)  # Phase should be within 10 degrees

    def test_decaycos(self):
        x = np.linspace(0, 10, 100)
        params = [1, 2, 0.5, 45, 3]  # y0, yscale, freq, phase, decay
        y = decaycos(x, *params)
        self.assertEqual(len(y), len(x))

        # Check if function decays properly
        last_oscillation_amp = abs(y[-1] - params[0])
        first_oscillation_amp = abs(y[0] - params[0])
        self.assertLess(last_oscillation_amp, first_oscillation_amp)

    def test_fitdecaycos(self):
        xdata = np.linspace(0, 10, 300)
        params = [1, 2, 0.5, 45, 3]  # y0, yscale, freq, phase, decay
        ydata = decaycos(xdata, *params) + np.random.normal(0, 0.1, len(xdata))

        fitparams, _ = fitdecaycos(xdata, ydata)

        # Check if the fitted parameters are close to the original ones
        self.assertAlmostEqual(fitparams[0], params[0], delta=0.3)  # y0
        self.assertAlmostEqual(fitparams[1], params[1], delta=0.5)  # yscale
        self.assertAlmostEqual(fitparams[2], params[2], delta=0.1)  # freq

        # For phase, handle wrapping around 360 degrees
        phase_diff = abs((fitparams[3] - params[3]) % 360)
        phase_diff = min(phase_diff, 360 - phase_diff)
        self.assertLess(phase_diff, 15)  # Phase should be within 15 degrees

        self.assertAlmostEqual(fitparams[4], params[4], delta=1.0)  # decay


if __name__ == "__main__":
    unittest.main()
