from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class WaveForm(ABC):
    @abstractmethod
    def __init__(self, pulse_cfg: Dict[str, Any]):
        pass

    @abstractmethod
    def numpy(self, num_sample: int) -> np.ndarray:
        pass


class ConstWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")

    def numpy(self, num_sample: int) -> np.ndarray:
        return np.ones(num_sample, dtype=complex)


class GaussWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        self.sigma = pulse_cfg["sigma"]

        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")
        if self.sigma <= 0:
            raise ValueError(f"無效的sigma值: {self.sigma}")

    def numpy(self, num_sample: int) -> np.ndarray:
        # 生成高斯波形，振幅為1
        x = np.linspace(-self.length / 2, self.length / 2, num_sample, dtype=complex)
        return np.exp(-0.5 * (x / self.sigma) ** 2)


class CosineWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]

        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")

    def numpy(self, num_sample: int) -> np.ndarray:
        # 生成餘弦波形，從0到2pi，振幅為1
        x = np.linspace(0, 2 * np.pi, num_sample, dtype=complex)
        return 0.5 * (1 - np.cos(x))


class DragWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        self.sigma = pulse_cfg["sigma"]
        self.alpha = pulse_cfg["alpha"]

        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")
        if self.sigma <= 0:
            raise ValueError(f"無效的sigma值: {self.sigma}")

    def numpy(self, num_sample: int) -> np.ndarray:
        # 生成DRAG波形，包含實部和虛部
        x = np.linspace(-self.length / 2, self.length / 2, num_sample, dtype=complex)
        gauss = np.exp(-0.5 * (x / self.sigma) ** 2)
        # DRAG修正項（虛部）
        deriv = -x / (self.sigma**2) * gauss

        # 返回的是複數波形
        return gauss + 1j * self.alpha * deriv


class FlatTopWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        raise_cfg = pulse_cfg["raise_pulse"]

        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")

        # 獲取上升/下降部分的波形類型和長度
        self.raise_style = raise_cfg["style"]
        self.raise_length = raise_cfg["length"]

        if self.raise_length * 2 >= self.length:
            raise ValueError(
                f"上升/下降部分太長: {self.raise_length}, 應小於波形總長的一半: {self.length / 2}"
            )

        # 創建上升/下降波形
        self.raise_waveform = make_waveform(raise_cfg)

    def numpy(self, num_sample: int) -> np.ndarray:
        # 計算平頂部分、上升和下降部分的點數
        raise_samples = int(self.raise_length / self.length * num_sample)
        if raise_samples * 2 >= num_sample:  # 確保平頂部分有點數
            raise_samples = num_sample // 2 - 1

        flat_samples = num_sample - 2 * raise_samples

        # 生成上升部分波形
        raise_wave = self.raise_waveform.numpy(2 * raise_samples)

        # 平頂部分為常數1
        flat_wave = np.ones(flat_samples, dtype=complex)

        raise_up_wave = raise_wave[:raise_samples]
        raise_down_wave = raise_wave[raise_samples:]

        # 組合完整波形
        return np.concatenate([raise_up_wave, flat_wave, raise_down_wave])


def make_waveform(pulse_cfg: Dict[str, Any]) -> WaveForm:
    style = pulse_cfg["style"]
    if style == "const":
        return ConstWaveForm(pulse_cfg)
    elif style == "gauss":
        return GaussWaveForm(pulse_cfg)
    elif style == "cosine":
        return CosineWaveForm(pulse_cfg)
    elif style == "drag":
        return DragWaveForm(pulse_cfg)
    elif style == "flat_top":
        return FlatTopWaveForm(pulse_cfg)
    else:
        raise ValueError(f"Unknown waveform style: {style}")
