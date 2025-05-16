import unittest

import numpy as np

from lib.zcu_tools.program.v2.base.simulate.waveform import (
    ConstWaveForm,
    CosineWaveForm,
    DragWaveForm,
    FlatTopWaveForm,
    GaussWaveForm,
    make_waveform,
)


class TestWaveForm(unittest.TestCase):
    def setUp(self):
        self.loop_dict = {"gain": 101, "freq": 30, "length": 11}

        self.SAMPLE_NUM = 100
        self.EXPECTED_SHAPE = (*self.loop_dict.values(), self.SAMPLE_NUM)

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
        times, wave = waveform.numpy(self.loop_dict, self.SAMPLE_NUM)

        self.assertEqual(times.shape, wave.shape)
        self.assertEqual(wave.shape, self.EXPECTED_SHAPE)
        self.assertTrue(np.allclose(wave, np.ones(self.EXPECTED_SHAPE, dtype=complex)))
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

        times, wave = waveform.numpy(self.loop_dict, self.SAMPLE_NUM)

        self.assertEqual(times.shape, wave.shape)
        self.assertEqual(wave.shape, self.EXPECTED_SHAPE)
        self.assertEqual(wave.dtype, np.complex128)

        # 檢查波形特性
        self.assertTrue(
            np.allclose(wave.real[..., self.SAMPLE_NUM // 2], 1.0, atol=0.01)
        )
        self.assertTrue(
            np.allclose(wave.imag[..., self.SAMPLE_NUM // 2], 0.0, atol=0.01)
        )
        self.assertTrue(np.allclose(wave.real[..., 0], wave.real[..., -1], atol=0.01))
        self.assertTrue(np.allclose(wave.imag[..., 0], wave.imag[..., -1], atol=0.01))

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

        times, wave = waveform.numpy(self.loop_dict, self.SAMPLE_NUM)

        self.assertEqual(times.shape, wave.shape)
        self.assertEqual(wave.shape, self.EXPECTED_SHAPE)
        self.assertEqual(wave.dtype, np.complex128)

        # 檢查波形特性
        self.assertTrue(
            np.allclose(wave.real[..., self.SAMPLE_NUM // 2], 1.0, atol=0.01)
        )
        self.assertTrue(
            np.allclose(wave.imag[..., self.SAMPLE_NUM // 2], 0.0, atol=0.01)
        )
        self.assertTrue(np.allclose(wave.real[..., 0], wave.real[..., -1], atol=0.01))
        self.assertTrue(np.allclose(wave.imag[..., 0], wave.imag[..., -1], atol=0.01))

    def test_drag_waveform_init(self):
        # 測試有效初始化
        pulse_cfg = {"length": 10, "sigma": 2, "alpha": 0.5, "delta": 1.0}
        waveform = DragWaveForm(pulse_cfg)
        self.assertEqual(waveform.length, 10)
        self.assertEqual(waveform.sigma, 2)
        self.assertEqual(waveform.alpha, 0.5)
        self.assertEqual(waveform.delta, 1.0)

        # 測試無效初始化
        with self.assertRaises(ValueError):
            DragWaveForm({"length": -5, "sigma": 2, "alpha": 0.5, "delta": 1.0})

        with self.assertRaises(ValueError):
            DragWaveForm({"length": 10, "sigma": -1, "alpha": 0.5, "delta": 1.0})

    def test_drag_waveform_numpy(self):
        pulse_cfg = {"length": 10, "sigma": 2, "alpha": 0.5, "delta": 1.0}
        waveform = DragWaveForm(pulse_cfg)

        times, wave = waveform.numpy(self.loop_dict, self.SAMPLE_NUM)

        self.assertEqual(times.shape, wave.shape)
        self.assertEqual(wave.shape, self.EXPECTED_SHAPE)
        self.assertEqual(wave.dtype, np.complex128)

        # 檢查波形特性
        # 實部應類似高斯波形，允許誤差0.01
        self.assertTrue(
            np.allclose(wave.real[..., self.SAMPLE_NUM // 2], 1.0, atol=0.01)
        )

        # 中心點的虛部應為0（導數為0）
        self.assertTrue(
            np.allclose(wave.imag[..., self.SAMPLE_NUM // 2], 0.0, atol=0.01)
        )

        # 檢查虛部在中心兩側的對稱性（符號相反）
        quarter = self.SAMPLE_NUM // 4
        self.assertTrue(
            np.allclose(
                wave.imag[..., self.SAMPLE_NUM // 2 - quarter],
                -wave.imag[..., self.SAMPLE_NUM // 2 + quarter],
                atol=0.01,
            )
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
                {"length": 10, "raise_pulse": {"style": "cosine", "length": 11}}
            )

    def test_flat_top_waveform_numpy(self):
        raise_cfg = {"style": "cosine", "length": 2}
        pulse_cfg = {"length": 10, "raise_pulse": raise_cfg}
        waveform = FlatTopWaveForm(pulse_cfg)

        times, wave = waveform.numpy(self.loop_dict, self.SAMPLE_NUM)

        self.assertEqual(times.shape, wave.shape)
        self.assertEqual(wave.shape, self.EXPECTED_SHAPE)
        self.assertEqual(wave.dtype, np.complex128)

        # 檢查波形特性
        # 前20%應該是上升部分
        rise_end = self.SAMPLE_NUM // 5
        # 後20%應該是下降部分
        fall_start = self.SAMPLE_NUM * 4 // 5

        # 檢查平頂部分是否為1
        flat_part = wave[..., rise_end:fall_start]
        self.assertTrue(np.allclose(flat_part, np.ones_like(flat_part)))

        # 檢查起始和結束值
        self.assertTrue(np.allclose(wave[..., 0], wave[..., -1]))
        self.assertTrue(np.all(np.abs(wave[..., 0]) < 0.1))

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
        drag_cfg = {
            "style": "drag",
            "length": 10,
            "sigma": 2,
            "alpha": 0.5,
            "delta": 1.0,
        }
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
