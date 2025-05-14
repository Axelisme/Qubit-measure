from typing import Any, Dict, List

import numpy as np

from .waveform import format_param, make_waveform


class Pulse:
    def __init__(self, start_t: float, pulse_cfg: Dict[str, Any]):
        self.ch = pulse_cfg["ch"]
        self.start_t = start_t
        self.gain = pulse_cfg["gain"]
        self.waveform = make_waveform(pulse_cfg)

    def get_signal(self, prog, times: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get the signal of the pulse at the given times using interpolation.
        Args:
            prog: MyProgramV2
            times: The times at which to get the signal.
        Returns:
            A dictionary of signals, where the key is the channel number and the value is the signal array.
        """

        start_t = format_param(prog, self.start_t)
        gain = format_param(prog, self.gain)

        num_samples = len(times)
        w_times, w_signals = self.waveform.numpy(prog, num_samples)
        w_times += start_t

        # Interpolate the waveform at last axis
        flat_times = np.reshape(w_times, (-1, num_samples))
        flat_signals = np.reshape(w_signals, (-1, num_samples))

        signals = np.array(
            [
                np.interp(times, flat_times[i], flat_signals[i], left=0.0, right=0.0)
                for i in range(flat_times.shape[0])
            ]
        ).reshape((*w_times.shape[:-1], len(times)))

        return {self.ch: gain[..., None] * signals}


def pulses_to_signal(
    prog, pulses: List[Pulse], times: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Convert a list of pulses to dictionary of signals. In the dictionary, the key is the channel number and the value is the signal array.
    """
    signal_dict: Dict[int, np.ndarray] = {}

    for pulse in pulses:
        pulse_signal_dict = pulse.get_signal(prog, times)
        for ch, signal in pulse_signal_dict.items():
            if ch not in signal_dict:
                signal_dict[ch] = signal
            else:
                # TODO: in reality, it should not be accumulate
                signal_dict[ch] += signal

    return signal_dict
