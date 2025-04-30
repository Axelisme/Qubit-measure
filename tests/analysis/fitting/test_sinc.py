import unittest

import numpy as np

from lib.zcu_tools.analysis.fitting.sinc import (
    fitsinc,
    sincfunc,
)


class TestSinc(unittest.TestCase):
    def test_sincfun(self):
        x = np.linspace(-5, 5, 200)
        params = [0.5, 0.2, 2.0, 1.0, 1.5]  # y0, slope, yscale, x0, gamma
        y = sincfunc(x, *params)
        self.assertEqual(len(y), len(x))

        # 峰值位置應接近 x0
        peak_index = np.argmax(y)
        self.assertAlmostEqual(x[peak_index], params[3], delta=0.1)

        # 峰值高度應為 y0 + yscale
        expected_peak = params[0] + params[2]
        self.assertAlmostEqual(y[peak_index], expected_peak, delta=0.1)

    def test_fitsinc(self):
        xdata = np.linspace(-5, 5, 200)
        params = [0.5, 0.2, 2.0, 1.0, 1.5]
        # 加入少量高斯噪聲
        ydata = sincfunc(xdata, *params) + np.random.normal(0, 0.05, len(xdata))

        fitparams, _ = fitsinc(xdata, ydata)

        # 檢查擬合參數用較寬容範圍
        self.assertAlmostEqual(fitparams[0], params[0], delta=0.3)  # y0
        self.assertAlmostEqual(fitparams[2], params[2], delta=0.3)  # yscale
        self.assertAlmostEqual(fitparams[3], params[3], delta=0.3)  # x0
        self.assertAlmostEqual(fitparams[4], params[4], delta=0.5)  # gamma


if __name__ == "__main__":
    unittest.main()
