from typing import Dict, Any
from abc import ABC, abstractmethod

import numpy as np


class WaveForm(ABC):
    @abstractmethod
    def __init__(self, pulse_cfg: Dict[str, Any]):
        pass

    @abstractmethod
    def numpy(self, num_sample: int) -> np.ndarray:
        pass


class ConstWaveForm(WaveForm):
    pass


class GaussWaveForm(WaveForm):
    pass


class CosineWaveForm(WaveForm):
    pass


class DragWaveForm(WaveForm):
    pass


class FlatTopWaveForm(WaveForm):
    pass


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
