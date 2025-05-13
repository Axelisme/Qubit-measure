from typing import Dict, Any, List

import numpy as np

from .waveform import make_waveform


class Pulse:
    def __init__(self, start_t: float, pulse_cfg: Dict[str, Any]):
        self.start_t = start_t
        self.ch = pulse_cfg["ch"]
        self.length = pulse_cfg["length"]
        self.waveform = make_waveform(pulse_cfg)

    def get_signal(self, times: np.ndarray) -> Dict[int, np.ndarray]:
        pass


def pulses_to_signal(pulses: List[Pulse], times: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Convert a list of pulses to dictionary of signals. In the dictionary, the key is the channel number and the value is the signal array.
    """
    pass
