from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from myqick.asm_v2 import QickParam


def format_param(prog, param: Any) -> np.ndarray:
    loop_dict = prog.loop_dict
    if isinstance(param, QickParam):
        values = param.start
        for name, count in loop_dict.items():
            if name in param.spans:
                span = param.spans[name]
                steps = np.linspace(0, span, count)
                values = np.add.outer(values, steps)
            else:
                values = np.add.outer(values, np.zeros(count))
        return values

    return np.full(tuple(loop_dict.values()), fill_value=param)


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
        if self.length < 0:
            raise ValueError(f"無效的波形長度: {self.length}")

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)

        times = np.linspace(0.0, length, num_sample, axis=-1)
        signals = np.ones_like(times, dtype=complex)

        return times, signals


class GaussWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        self.sigma = pulse_cfg["sigma"]

        if self.length < 0:
            raise ValueError(f"無效的波形長度: {self.length}")
        if self.sigma < 0:
            raise ValueError(f"無效的sigma值: {self.sigma}")

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)
        sigma = format_param(prog, self.sigma)

        # 生成高斯波形，振幅為1
        times = np.linspace(0.0, length, num_sample, axis=-1)
        x = times - length[..., None] / 2
        signals = np.exp(-0.5 * (x / sigma[..., None]) ** 2)
        signals = signals.astype(complex)

        return times, signals


class CosineWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]

        if self.length < 0:
            raise ValueError(f"無效的波形長度: {self.length}")

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)

        # 生成餘弦波形，從0到2pi，振幅為1
        times = np.linspace(0.0, length, num_sample, axis=-1)
        signals = 0.5 * (1 - np.cos(2 * np.pi * times / length[..., None]))
        signals = signals.astype(complex)

        return times, signals


class DragWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        self.sigma = pulse_cfg["sigma"]
        self.alpha = pulse_cfg["alpha"]
        self.delta = pulse_cfg["delta"]

        if self.length < 0:
            raise ValueError(f"無效的波形長度: {self.length}")
        if self.sigma < 0:
            raise ValueError(f"無效的sigma值: {self.sigma}")

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)
        sigma = format_param(prog, self.sigma)
        alpha = format_param(prog, self.alpha)
        delta = format_param(prog, self.delta)
        times = np.linspace(0.0, length, num_sample, axis=-1)

        x = times - length[..., None] / 2
        gauss = np.exp(-0.5 * (x / sigma[..., None]) ** 2)
        deriv = -x / (sigma[..., None] ** 2) * gauss
        signals = gauss - 1j * alpha[..., None] * deriv / delta[..., None]

        return times, signals


class FlatTopWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        raise_cfg = pulse_cfg["raise_pulse"]

        if self.length < 0:
            raise ValueError(f"無效的波形長度: {self.length}")

        # 獲取上升/下降部分的波形類型和長度
        self.raise_style = raise_cfg["style"]
        self.raise_length = raise_cfg["length"]

        if self.raise_length * 2 > self.length:
            raise ValueError(
                f"上升/下降部分太長: {self.raise_length}, 應小於波形總長的一半: {self.length / 2}"
            )

        # 創建上升/下降波形
        self.raise_waveform = make_waveform(raise_cfg)

    def numpy(self, prog, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(prog, self.length)
        raise_length = format_param(prog, self.raise_length)

        times = np.linspace(0.0, length, num_sample, axis=-1)

        # 計算平頂部分、上升和下降部分的點數
        raise_nums = (0.5 * raise_length / length * num_sample).astype(int)
        raise_nums = np.clip(raise_nums, 0, None)

        uni_nums, num_groups = np.unique(raise_nums.ravel(), return_inverse=True)
        flat_signals = np.ones((raise_nums.size, num_sample), dtype=complex)
        for i, num in enumerate(uni_nums):
            _, raise_wave = self.raise_waveform.numpy(prog, 2 * num)
            flat_raise_wave = raise_wave.reshape(-1, 2 * num)

            num_idx = np.flatnonzero(num_groups == i)
            flat_signals[num_idx, :num] = flat_raise_wave[num_idx, :num]
            flat_signals[num_idx, -num:] = flat_raise_wave[num_idx, -num:]
        signals = flat_signals.reshape(*raise_nums.shape, num_sample)

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
