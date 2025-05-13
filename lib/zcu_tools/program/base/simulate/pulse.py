from copy import deepcopy
from typing import Any, Dict, List

import numpy as np

from .waveform import make_waveform


class Pulse:
    def __init__(self, start_t: float, pulse_cfg: Dict[str, Any]):
        self.start_t = start_t
        self.ch = pulse_cfg["ch"]
        self.length = pulse_cfg["length"]
        self.waveform = make_waveform(pulse_cfg)

    def get_signal(self, times: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get the signal of the pulse at the given times using interpolation.
        Args:
            times: The times at which to get the signal.
        Returns:
            A dictionary of signals, where the key is the channel number and the value is the signal array.
        """
        num_samples = max(times.size, 513)
        waveform_times = np.linspace(0, self.length, num_samples, endpoint=True)
        waveform_signal = self.waveform.numpy(num_samples)

        # use interpolation to get the signal at the given times
        return {
            self.ch: np.interp(
                times - self.start_t, waveform_times, waveform_signal, left=0, right=0
            )
        }


def pulses_to_signal(pulses: List[Pulse], times: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Convert a list of pulses to dictionary of signals. In the dictionary, the key is the channel number and the value is the signal array.
    """
    signals_by_channel: Dict[int, np.ndarray] = {}

    # sort pulses by start_t
    pulses = deepcopy(pulses)
    pulses.sort(key=lambda x: x.start_t)

    for pulse in pulses:
        pulse_signal_dict = pulse.get_signal(times)
        ch, signal_array = list(pulse_signal_dict.items())[0]
        if ch not in signals_by_channel:
            signals_by_channel[ch] = signal_array
        else:
            signals_by_channel[ch] += signal_array

    return signals_by_channel
