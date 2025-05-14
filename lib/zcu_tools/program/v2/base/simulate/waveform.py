from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from myqick.asm_v2 import QickParam


def format_param(prog, param: Any) -> np.ndarray:
    if isinstance(param, QickParam):
        return param.to_array(prog.loop_dict, all_loops=True)

    return np.full((1,) * len(prog.loop_dict), fill_value=param)


class WaveForm(ABC):
    @abstractmethod
    def __init__(self, pulse_cfg: Dict[str, Any]):
        pass

    @abstractmethod
    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        pass


class ConstWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)

        times = np.linspace(0.0, length, num_sample)
        signals = np.ones_like(times, dtype=complex)

        return times, signals


class GaussWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        self.sigma = pulse_cfg["sigma"]

        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")
        if self.sigma <= 0:
            raise ValueError(f"無效的sigma值: {self.sigma}")

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)
        sigma = format_param(prog, self.sigma)

        # 生成高斯波形，振幅為1
        times = np.linspace(0.0, length, num_sample)
        signals = np.exp(-0.5 * ((times - length / 2) / sigma[..., None]) ** 2)
        signals = signals.astype(complex)

        return times, signals


class CosineWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]

        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)

        # 生成餘弦波形，從0到2pi，振幅為1
        times = np.linspace(0.0, length, num_sample)
        signals = 0.5 * (1 - np.cos(2 * np.pi * times / length))
        signals = signals.astype(complex)

        return times, signals


class DragWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        self.sigma = pulse_cfg["sigma"]
        self.alpha = pulse_cfg["alpha"]

        if self.length <= 0:
            raise ValueError(f"無效的波形長度: {self.length}")
        if self.sigma <= 0:
            raise ValueError(f"無效的sigma值: {self.sigma}")

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)
        sigma = format_param(prog, self.sigma)
        alpha = format_param(prog, self.alpha)

        times = np.linspace(0.0, length, num_sample)

        x = times - length / 2
        gauss = np.exp(-0.5 * (x / sigma[..., None]) ** 2)
        deriv = -x / (sigma[..., None] ** 2) * gauss
        signals = gauss + 1j * alpha[..., None] * deriv

        return times, signals


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

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)
        raise_length = format_param(prog, self.raise_length)

        # 計算平頂部分、上升和下降部分的點數
        raise_samples = int(raise_length / length * num_sample)
        raise_samples = np.clip(raise_samples, 0, None)

        flat_samples = num_sample - 2 * raise_samples

        times = np.linspace(0.0, length, num_sample)

        # 生成上升部分波形
        _, raise_wave = self.raise_waveform.numpy(prog, 2 * raise_samples)

        raise_up_wave = raise_wave[..., :raise_samples]
        flat_wave = np.ones((*length.shape, flat_samples), dtype=complex)
        raise_down_wave = raise_wave[..., raise_samples:]
        signals = np.concatenate([raise_up_wave, flat_wave, raise_down_wave], axis=-1)

        return times, signals


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
