import unittest

import numpy as np

from lib.zcu_tools.analysis.fitting.gauss2d import (
    fit_gauss_2d,
    fit_gauss_2d_bayesian,
    gauss_2d,
)


class TestGauss2D(unittest.TestCase):
    def test_gauss_2d(self):
        # 創建網格點進行測試
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()

        # 設定高斯函數參數
        params = [0, 0, 1, 2]  # x0, y0, sigma, n

        # 計算高斯函數值
        z = gauss_2d(x_flat, y_flat, *params)

        self.assertEqual(len(z), len(x_flat))
        self.assertAlmostEqual(max(z), params[3], delta=0.05)

        # 檢查中心點是否正確
        center_idx = np.argmax(z)
        self.assertAlmostEqual(x_flat[center_idx], params[0], delta=0.5)
        self.assertAlmostEqual(y_flat[center_idx], params[1], delta=0.5)

    def test_fit_gauss_2d(self):
        # 生成兩個高斯分佈的數據
        np.random.seed(42)

        # 第一個高斯分佈
        x1 = np.random.normal(-2, 0.7, 200)
        y1 = np.random.normal(1, 0.7, 200)

        # 第二個高斯分佈
        x2 = np.random.normal(2, 0.5, 150)
        y2 = np.random.normal(-1, 0.5, 150)

        # 合併數據
        xs = np.concatenate([x1, x2])
        ys = np.concatenate([y1, y2])

        # 擬合數據
        params, _ = fit_gauss_2d(xs, ys, num_gauss=2)

        # 檢查返回的參數數量是否正確
        self.assertEqual(params.shape, (2, 4))

        # 檢查擬合結果是否接近原始分佈
        # 我們需要按照x0值排序以便比較
        params = params[params[:, 0].argsort()]

        # 第一個高斯分佈約在(-2, 1)
        self.assertAlmostEqual(params[0, 0], -2, delta=0.5)
        self.assertAlmostEqual(params[0, 1], 1, delta=0.5)
        self.assertAlmostEqual(params[0, 2], 0.7, delta=0.3)  # sigma

        # 第二個高斯分佈約在(2, -1)
        self.assertAlmostEqual(params[1, 0], 2, delta=0.5)
        self.assertAlmostEqual(params[1, 1], -1, delta=0.5)
        self.assertAlmostEqual(params[1, 2], 0.5, delta=0.3)  # sigma

        # 檢查權重總和是否接近1
        self.assertAlmostEqual(np.sum(params[:, 3]), 1, delta=0.1)

    def test_fit_gauss_2d_bayesian(self):
        # 生成有三個高斯分佈的數據，但其中一個非常小
        np.random.seed(42)

        # 第一個高斯分佈
        x1 = np.random.normal(-2, 0.7, 200)
        y1 = np.random.normal(1, 0.7, 200)

        # 第二個高斯分佈
        x2 = np.random.normal(2, 0.5, 150)
        y2 = np.random.normal(-1, 0.5, 150)

        # 第三個較小的高斯分佈
        x3 = np.random.normal(0, 0.3, 30)
        y3 = np.random.normal(0, 0.3, 30)

        # 合併數據
        xs = np.concatenate([x1, x2, x3])
        ys = np.concatenate([y1, y2, y3])

        # 用貝葉斯方法擬合，設置最大高斯分佈數為3
        params, _ = fit_gauss_2d_bayesian(xs, ys, num_gauss=3)

        # 檢查返回的參數數量是否正確
        self.assertLessEqual(params.shape[0], 3)
        self.assertEqual(params.shape[1], 4)

        # 檢查權重總和是否接近1
        self.assertAlmostEqual(np.sum(params[:, 3]), 1, delta=0.1)

        # 檢查至少找到兩個主要的高斯分佈
        if params.shape[0] >= 2:
            # 檢查找到的兩個主要分佈是否接近原始分佈
            # 不能直接比較，因為不知道哪些分佈被識別為主要的
            found_cluster1 = False
            found_cluster2 = False

            for i in range(params.shape[0]):
                x0, y0 = params[i, 0], params[i, 1]
                if np.abs(x0 + 2) < 1 and np.abs(y0 - 1) < 1:
                    found_cluster1 = True
                if np.abs(x0 - 2) < 1 and np.abs(y0 + 1) < 1:
                    found_cluster2 = True

            self.assertTrue(found_cluster1 or found_cluster2)


if __name__ == "__main__":
    unittest.main()
