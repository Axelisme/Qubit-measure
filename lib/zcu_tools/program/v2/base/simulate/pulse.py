from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np

from .waveform import format_param, make_waveform


class Pulse:
    def __init__(self, start_t: float, pulse_cfg: Dict[str, Any]):
        self.ch = pulse_cfg["ch"]
        self.start_t = start_t
        self.gain = pulse_cfg["gain"]
        self.waveform = make_waveform(pulse_cfg)

    def get_signal(self, loop_dict, times: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get the signal of the pulse at the given times using interpolation.
        Args:
            loop_dict: loop_dict
            times: The times at which to get the signal.
        Returns:
            A dictionary of signals, where the key is the channel number and the value is the signal array.
        """

        start_t = format_param(loop_dict, self.start_t)
        gain = format_param(loop_dict, self.gain)

        num_samples = len(times)
        w_times, w_signals = self.waveform.numpy(loop_dict, num_samples)
        w_times += start_t[..., None]

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


def visualize_pulse(
    pulse_cfgs: Union[Dict[str, Any], List[Dict[str, Any]]], time_fly: float = 0.0
):
    import matplotlib.pyplot as plt

    pulse_cfgs = deepcopy(pulse_cfgs)

    if not isinstance(pulse_cfgs, list):
        pulse_cfgs = [pulse_cfgs]

    max_length = 0.1
    for pulse_cfg in pulse_cfgs:
        pulse_cfg.setdefault("ch", 0)
        pulse_cfg.setdefault("gain", 1.0)

        max_length = max(
            max_length,
            pulse_cfg.get("t", 0.0)
            + pulse_cfg["length"]
            + pulse_cfg.get("post_delay", 0.0),
        )

    max_gain = 0.0
    min_gain = 0.0
    loop_dict = {}
    times = np.linspace(0.0, max_length, 3001)
    for pulse_cfg in pulse_cfgs:
        pulse = Pulse(0.0, pulse_cfg)

        signal_dict = pulse.get_signal(loop_dict, times)
        for ch, signal in signal_dict.items():
            plt.plot(times, signal.real, label=f"ch {ch} real")
            plt.plot(times, signal.imag, label=f"ch {ch} imag")
            max_gain = max(max_gain, np.max(signal.real), np.max(signal.imag))
            min_gain = min(min_gain, np.min(signal.real), np.min(signal.imag))

        if "trig_offset" in pulse_cfg:
            offset = pulse_cfg["trig_offset"] + pulse_cfg.get("pre_delay", 0.0)
            plt.axvline(offset - time_fly, color="red", linestyle="--")
            if "ro_length" in pulse_cfg:
                ro_length = pulse_cfg["ro_length"]
                plt.axvline(offset + ro_length - time_fly, color="red", linestyle="--")

    plt.legend()
    plt.xlabel("Time (us)")
    plt.ylabel("I/Q")
    plt.ylim(min_gain - 0.01, max_gain + 0.01)
    plt.xlim(0.0, max_length)
    plt.grid(True)
    plt.show()


def pulses_to_signal(
    loop_dict, pulses: List[Pulse], times: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Convert a list of pulses to dictionary of signals. In the dictionary, the key is the channel number and the value is the signal array.
    """
    signal_dict: Dict[int, np.ndarray] = {}

    for pulse in pulses:
        pulse_signal_dict = pulse.get_signal(loop_dict, times)
        for ch, signal in pulse_signal_dict.items():
            if ch not in signal_dict:
                signal_dict[ch] = signal
            else:
                # TODO: in reality, it should not be accumulate
                signal_dict[ch] += signal

    return signal_dict
