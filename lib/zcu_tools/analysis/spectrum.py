from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict


def format_rawdata(
    dev_values: NDArray[np.float64],
    freqs: NDArray[np.float64],  # in Hz
    signals: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    freqs = freqs / 1e9  # convert to GHz

    if dev_values[0] > dev_values[-1]:  # Ensure that the fluxes are in increasing
        dev_values = dev_values[::-1]
        signals = signals[::-1, :]
    if freqs[0] > freqs[-1]:  # Ensure that the frequencies are in increasing
        freqs = freqs[::-1]
        signals = signals[:, ::-1]

    return dev_values, freqs, signals


class SpectrumData(TypedDict):
    dev_values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
