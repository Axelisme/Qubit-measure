from typing import List, Optional

import numpy as np
from numpy import ndarray


def calc_snr(avg_d: ndarray, std_d: ndarray) -> ndarray:
    # avg_d: (ge, *sweep)
    # std_d: (ge, *sweep)

    contrast = avg_d[1, ...] - avg_d[0, ...]  # (*sweep)
    noise2_i = np.sum(std_d.real**2, axis=0)  # (*sweep)
    noise2_q = np.sum(std_d.imag**2, axis=0)  # (*sweep)
    noise = np.sqrt(noise2_i * contrast.real**2 + noise2_q * contrast.imag**2) / np.abs(
        contrast
    )

    return contrast / noise


def snr_as_signal(
    ir: int, avg_d: List[ndarray], std_d: Optional[List[ndarray]]
) -> np.ndarray:
    assert std_d is not None, "std_d should not be None"

    avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
    std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

    return calc_snr(avg_s, std_s)
