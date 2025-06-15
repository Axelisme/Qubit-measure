from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from myqick.asm_v2 import QickParam


def format_param(loop_dict: Dict[str, int], param: Any) -> np.ndarray:
    if isinstance(param, QickParam):
        values = param.start
        for name, count in loop_dict.items():
            if name in param.spans:
                span = param.spans[name]
                steps = np.linspace(0, span, count)
                values = np.add.outer(values, steps)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                values = np.add.outer(values, np.zeros(count))  # pyright: ignore[reportAttributeAccessIssue]
        return values

    return np.full(tuple(loop_dict.values()), fill_value=param)


class WaveForm(ABC):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        self.length = pulse_cfg["length"]
        self.phase = pulse_cfg.get("phase", 0.0)  # raise pulse don't need this
        if self.length < 0 or self.length == 0:
            raise ValueError(f"無效的波形長度: {self.length}")

    def get_phase(self, loop_dict) -> np.ndarray:
        phase = format_param(loop_dict, self.phase)
        return np.exp(1j * phase[..., None] / 180 * np.pi)  # type: ignore

    @abstractmethod
    def numpy(self, loop_dict, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        pass


class ConstWaveForm(WaveForm):
    def numpy(self, loop_dict, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(loop_dict, self.length)

        times = np.linspace(0.0, length, num_sample, axis=-1)
        signals = np.ones_like(times, dtype=complex)

        return times, signals * self.get_phase(loop_dict)


class GaussWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        super().__init__(pulse_cfg)
        self.sigma = pulse_cfg["sigma"]

        if self.sigma < 0:
            raise ValueError(f"無效的sigma值: {self.sigma}")

    def numpy(self, loop_dict, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(loop_dict, self.length)
        sigma = format_param(loop_dict, self.sigma)

        # 生成高斯波形，振幅為1
        times = np.linspace(0.0, length, num_sample, axis=-1)
        x = times - length[..., None] / 2
        signals = np.exp(-0.5 * (x / sigma[..., None]) ** 2)  # type: ignore

        return times, signals * self.get_phase(loop_dict)


class CosineWaveForm(WaveForm):
    def numpy(self, loop_dict, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(loop_dict, self.length)

        # 生成餘弦波形，從0到2pi，振幅為1
        times = np.linspace(0.0, length, num_sample, axis=-1)
        signals = 0.5 * (1 - np.cos(2 * np.pi * times / length[..., None]))  # type: ignore

        return times, signals * self.get_phase(loop_dict)


class DragWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        super().__init__(pulse_cfg)
        self.sigma = pulse_cfg["sigma"]
        self.alpha = pulse_cfg["alpha"]
        self.delta = pulse_cfg["delta"]

        if self.sigma < 0:
            raise ValueError(f"無效的sigma值: {self.sigma}")

    def numpy(self, loop_dict, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(loop_dict, self.length)
        sigma = format_param(loop_dict, self.sigma)
        alpha = format_param(loop_dict, self.alpha)
        delta = format_param(loop_dict, self.delta)
        times = np.linspace(0.0, length, num_sample, axis=-1)

        x = times - length[..., None] / 2
        gauss = np.exp(-0.5 * (x / sigma[..., None]) ** 2)  # type: ignore
        deriv = -x / (sigma[..., None] ** 2) * gauss
        signals = gauss - 1j * alpha[..., None] * deriv / delta[..., None]

        return times, signals * self.get_phase(loop_dict)


class FlatTopWaveForm(WaveForm):
    def __init__(self, pulse_cfg: Dict[str, Any]):
        super().__init__(pulse_cfg)
        raise_cfg = pulse_cfg["raise_pulse"]

        # 獲取上升/下降部分的波形類型和長度
        self.raise_style = raise_cfg["style"]
        self.raise_length = raise_cfg["length"]

        if self.raise_length > self.length:
            raise ValueError(
                f"上升/下降部分太長: {self.raise_length}, 應小於波形總長的一半: {self.length / 2}"
            )

        # 創建上升/下降波形
        self.raise_waveform = make_waveform(raise_cfg)

    def numpy(self, loop_dict, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
        length = format_param(loop_dict, self.length)
        raise_length = format_param(loop_dict, self.raise_length)

        times = np.linspace(0.0, length, num_sample, axis=-1)

        # 計算平頂部分、上升和下降部分的點數
        raise_nums = (0.5 * raise_length / length * num_sample).astype(int)
        raise_nums = np.clip(raise_nums, 0, None)

        uni_nums, num_groups = np.unique(raise_nums.ravel(), return_inverse=True)
        flat_signals = np.ones((raise_nums.size, num_sample), dtype=complex)
        for i, num in enumerate(uni_nums):
            if num == 0:
                continue
            _, raise_wave = self.raise_waveform.numpy(loop_dict, 2 * num)
            flat_raise_wave = raise_wave.reshape(-1, 2 * num)

            num_idx = np.flatnonzero(num_groups == i)
            flat_signals[num_idx, :num] = flat_raise_wave[num_idx, :num]
            flat_signals[num_idx, -num:] = flat_raise_wave[num_idx, -num:]
        signals = flat_signals.reshape(*raise_nums.shape, num_sample)

        return times, signals * self.get_phase(loop_dict)


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
