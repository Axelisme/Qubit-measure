import unittest

import numpy as np

from lib.zcu_tools.program.v2.base.simulate.pulse import Pulse, pulses_to_signal


class TestPulse(unittest.TestCase):
    def setUp(self):
        # 預設測試時間軸
        self.times = np.linspace(0, 10, 100)
        self.loop_dict = {}

    def test_pulse_const_waveform(self):
        pulse_cfg = {"ch": 0, "length": 10, "style": "const", "gain": 1.0}
        pulse = Pulse(0, pulse_cfg)
        signal = pulse.get_signal(self.loop_dict, self.times)
        # get_signal應回傳np.ndarray
        self.assertIsInstance(signal, dict)
        self.assertIn(0, signal)
        self.assertIsInstance(signal[0], np.ndarray)
        self.assertEqual(signal[0].shape, self.times.shape)
        # 常數波形應全為1
        self.assertTrue(np.allclose(signal[0], 1.0))

    def test_pulse_gauss_waveform(self):
        pulse_cfg = {"ch": 1, "length": 10, "style": "gauss", "sigma": 2, "gain": 1.0}
        pulse = Pulse(0, pulse_cfg)
        signal = pulse.get_signal(self.loop_dict, self.times)

        self.assertIsInstance(signal, dict)
        self.assertIn(1, signal)
        self.assertIsInstance(signal[1], np.ndarray)
        self.assertEqual(signal[1].shape, self.times.shape)
        # 高斯波形最大值應大於邊緣
        self.assertGreater(np.max(np.abs(signal[1])), np.abs(signal[1][0]))
        self.assertGreater(np.max(np.abs(signal[1])), np.abs(signal[1][-1]))

    def test_pulse_with_start_time(self):
        # 測試start_t偏移
        pulse_cfg = {"ch": 0, "length": 5, "style": "const", "gain": 1.0}
        pulse = Pulse(2, pulse_cfg)
        signal = pulse.get_signal(self.loop_dict, self.times)
        self.assertIn(0, signal)
        self.assertIsInstance(signal[0], np.ndarray)
        self.assertEqual(signal[0].shape, self.times.shape)

        # 前2的時間應為0
        self.assertTrue(np.all(signal[0][self.times < 2] == 0))
        # 2~7之間應為1
        mask = (self.times >= 2) & (self.times <= 7)
        self.assertTrue(np.allclose(signal[0][mask], 1, atol=0.01))
        # 7之後應為0
        self.assertTrue(np.all(signal[0][self.times > 7] == 0))

    def test_pulses_to_signal_single_channel(self):
        # 兩個同通道pulse，訊號應疊加
        pulse1 = Pulse(0, {"ch": 0, "length": 10, "style": "const", "gain": 1.0})
        pulse2 = Pulse(0, {"ch": 0, "length": 10, "style": "const", "gain": 1.0})
        signals = pulses_to_signal(self.loop_dict, [pulse1, pulse2], self.times)
        self.assertIn(0, signals)
        self.assertTrue(np.allclose(signals[0], 2))

    def test_pulses_to_signal_multi_channel(self):
        # 不同通道pulse
        pulse1 = Pulse(0, {"ch": 0, "length": 10, "style": "const", "gain": 1.0})
        pulse2 = Pulse(0, {"ch": 1, "length": 10, "style": "const", "gain": 1.0})
        signals = pulses_to_signal(self.loop_dict, [pulse1, pulse2], self.times)
        self.assertIn(0, signals)
        self.assertIn(1, signals)
        self.assertTrue(np.allclose(signals[0], 1))
        self.assertTrue(np.allclose(signals[1], 1))

    def test_pulses_to_signal_overlap(self):
        # 測試部分重疊的pulse
        pulse1 = Pulse(0, {"ch": 0, "length": 5, "style": "const", "gain": 1.0})
        pulse2 = Pulse(3, {"ch": 0, "length": 5, "style": "const", "gain": 1.0})
        signals = pulses_to_signal(self.loop_dict, [pulse1, pulse2], self.times)

        # 0~3: 只有pulse1，3~5: 疊加，5~8: 只有pulse2
        self.assertTrue(np.allclose(signals[0][self.times < 3], 1, atol=0.01))
        overlap_mask = (self.times >= 3) & (self.times <= 5)
        self.assertTrue(np.allclose(signals[0][overlap_mask], 2, atol=0.01))
        self.assertTrue(
            np.allclose(signals[0][(self.times > 5) & (self.times < 8)], 1, atol=0.01)
        )
        self.assertTrue(np.all(signals[0][self.times >= 8] == 0))


if __name__ == "__main__":
    unittest.main()
