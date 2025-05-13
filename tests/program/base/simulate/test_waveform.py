import unittest

import numpy as np

from lib.zcu_tools.program.base.simulate.waveform import (
    ConstWaveForm,
    CosineWaveForm,
    DragWaveForm,
    FlatTopWaveForm,
    GaussWaveForm,
    make_waveform,
)


class TestWaveForm(unittest.TestCase):
    def test_const_waveform_init(self):
        # 測試有效初始化
        pulse_cfg = {"length": 10}
        waveform = ConstWaveForm(pulse_cfg)
        self.assertEqual(waveform.length, 10)

        # 測試無效初始化
        with self.assertRaises(ValueError):
            ConstWaveForm({"length": -5})

        with self.assertRaises(ValueError):
            ConstWaveForm({"length": 0})

    def test_const_waveform_numpy(self):
        pulse_cfg = {"length": 10}
        waveform = ConstWaveForm(pulse_cfg)

        # 測試生成的波形
        samples = 100
        wave = waveform.numpy(samples)

        self.assertEqual(len(wave), samples)
        self.assertTrue(np.allclose(wave, np.ones(samples, dtype=complex)))
        self.assertEqual(wave.dtype, np.complex128)

    def test_gauss_waveform_init(self):
        # 測試有效初始化
        pulse_cfg = {"length": 10, "sigma": 2}
        waveform = GaussWaveForm(pulse_cfg)
        self.assertEqual(waveform.length, 10)
        self.assertEqual(waveform.sigma, 2)

        # 測試無效初始化
        with self.assertRaises(ValueError):
            GaussWaveForm({"length": -5, "sigma": 2})

        with self.assertRaises(ValueError):
            GaussWaveForm({"length": 10, "sigma": -1})

    def test_gauss_waveform_numpy(self):
        pulse_cfg = {"length": 10, "sigma": 2}
        waveform = GaussWaveForm(pulse_cfg)

        samples = 100
        wave = waveform.numpy(samples)

        self.assertEqual(len(wave), samples)
        self.assertEqual(wave.dtype, np.complex128)

        # 檢查波形特性
        # 中間點允許誤差0.01
        self.assertAlmostEqual(abs(wave[samples // 2]), 1.0, delta=0.01)  # 中間點應為1
        self.assertLess(abs(wave[0]), abs(wave[samples // 2]))  # 邊緣值應小於中間值
        self.assertLess(abs(wave[-1]), abs(wave[samples // 2]))

    def test_cosine_waveform_init(self):
        # 測試有效初始化
        pulse_cfg = {"length": 10}
        waveform = CosineWaveForm(pulse_cfg)
        self.assertEqual(waveform.length, 10)

        # 測試無效初始化
        with self.assertRaises(ValueError):
            CosineWaveForm({"length": -5})

    def test_cosine_waveform_numpy(self):
        pulse_cfg = {"length": 10}
        waveform = CosineWaveForm(pulse_cfg)

        samples = 101
        wave = waveform.numpy(samples)

        self.assertEqual(len(wave), samples)
        self.assertEqual(wave.dtype, np.complex128)

        # 檢查波形特性
        self.assertAlmostEqual(abs(wave[0]), 0.0, places=6)  # 起始點應為0
        self.assertAlmostEqual(abs(wave[samples // 2]), 1.0, places=6)  # 中間點應為1
        self.assertAlmostEqual(abs(wave[-1]), 0.0, places=6)  # 結束點應為0
        max_index = np.argmax(np.abs(wave))
        self.assertEqual(max_index, samples // 2)  # 最大值應在中間

    def test_drag_waveform_init(self):
        # 測試有效初始化
        pulse_cfg = {"length": 10, "sigma": 2, "alpha": 0.5}
        waveform = DragWaveForm(pulse_cfg)
        self.assertEqual(waveform.length, 10)
        self.assertEqual(waveform.sigma, 2)
        self.assertEqual(waveform.alpha, 0.5)

        # 測試無效初始化
        with self.assertRaises(ValueError):
            DragWaveForm({"length": -5, "sigma": 2, "alpha": 0.5})

        with self.assertRaises(ValueError):
            DragWaveForm({"length": 10, "sigma": -1, "alpha": 0.5})

    def test_drag_waveform_numpy(self):
        pulse_cfg = {"length": 10, "sigma": 2, "alpha": 0.5}
        waveform = DragWaveForm(pulse_cfg)

        samples = 100
        wave = waveform.numpy(samples)

        self.assertEqual(len(wave), samples)
        self.assertEqual(wave.dtype, np.complex128)

        # 檢查波形特性
        # 實部應類似高斯波形，允許誤差0.01
        self.assertAlmostEqual(abs(wave.real[samples // 2]), 1.0, delta=0.01)

        # 中心點的虛部應為0（導數為0）
        self.assertAlmostEqual(wave.imag[samples // 2], 0.0, delta=0.01)

        # 檢查虛部在中心兩側的對稱性（符號相反）
        quarter = samples // 4
        self.assertAlmostEqual(
            wave.imag[samples // 2 - quarter],
            -wave.imag[samples // 2 + quarter],
            delta=0.01,
        )

    def test_flat_top_waveform_init(self):
        # 測試有效初始化
        raise_cfg = {"style": "cosine", "length": 2}
        pulse_cfg = {"length": 10, "raise_pulse": raise_cfg}
        waveform = FlatTopWaveForm(pulse_cfg)
        self.assertEqual(waveform.length, 10)
        self.assertEqual(waveform.raise_length, 2)

        # 測試無效初始化 - 總長度無效
        with self.assertRaises(ValueError):
            FlatTopWaveForm({"length": -5, "raise_pulse": raise_cfg})

        # 測試無效初始化 - 上升/下降部分太長
        with self.assertRaises(ValueError):
            FlatTopWaveForm(
                {"length": 10, "raise_pulse": {"style": "cosine", "length": 6}}
            )

    def test_flat_top_waveform_numpy(self):
        raise_cfg = {"style": "cosine", "length": 2}
        pulse_cfg = {"length": 10, "raise_pulse": raise_cfg}
        waveform = FlatTopWaveForm(pulse_cfg)

        samples = 100
        wave = waveform.numpy(samples)

        self.assertEqual(len(wave), samples)
        self.assertEqual(wave.dtype, np.complex128)

        # 檢查波形特性
        # 前20%應該是上升部分
        rise_end = samples // 5
        # 後20%應該是下降部分
        fall_start = samples * 4 // 5

        # 檢查平頂部分是否為1
        flat_part = wave[rise_end:fall_start]
        self.assertTrue(np.allclose(flat_part, np.ones_like(flat_part)))

        # 檢查起始和結束值
        self.assertLess(abs(wave[0]), 0.1)  # 起始值應小於0.1
        self.assertLess(abs(wave[-1]), 0.1)  # 結束值應小於0.1

    def test_make_waveform(self):
        # 測試常數波形
        const_cfg = {"style": "const", "length": 10}
        waveform = make_waveform(const_cfg)
        self.assertIsInstance(waveform, ConstWaveForm)

        # 測試高斯波形
        gauss_cfg = {"style": "gauss", "length": 10, "sigma": 2}
        waveform = make_waveform(gauss_cfg)
        self.assertIsInstance(waveform, GaussWaveForm)

        # 測試餘弦波形
        cosine_cfg = {"style": "cosine", "length": 10}
        waveform = make_waveform(cosine_cfg)
        self.assertIsInstance(waveform, CosineWaveForm)

        # 測試DRAG波形
        drag_cfg = {"style": "drag", "length": 10, "sigma": 2, "alpha": 0.5}
        waveform = make_waveform(drag_cfg)
        self.assertIsInstance(waveform, DragWaveForm)

        # 測試平頂波形
        raise_cfg = {"style": "cosine", "length": 2}
        flat_top_cfg = {"style": "flat_top", "length": 10, "raise_pulse": raise_cfg}
        waveform = make_waveform(flat_top_cfg)
        self.assertIsInstance(waveform, FlatTopWaveForm)

        # 測試未知波形類型
        with self.assertRaises(ValueError):
            make_waveform({"style": "unknown", "length": 10})


if __name__ == "__main__":
    unittest.main()
