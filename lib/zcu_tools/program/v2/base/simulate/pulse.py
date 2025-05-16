from typing import Any, Dict, List

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


def visualize_pulse(pulse_cfg: Dict[str, Any]):
    import matplotlib.pyplot as plt

    pulse = Pulse(0.0, pulse_cfg)
    loop_dict = {}
    times = np.linspace(0.0, pulse_cfg["length"], 1001)
    signal_dict = pulse.get_signal(loop_dict, times)
    for ch, signal in signal_dict.items():
        plt.plot(times, signal.real, label=f"ch {ch} real")
        plt.plot(times, signal.imag, label=f"ch {ch} imag")
    plt.legend()
    plt.xlabel("Time (us)")
    plt.ylabel("I/Q")
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
