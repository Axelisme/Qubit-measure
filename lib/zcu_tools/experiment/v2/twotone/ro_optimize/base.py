from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def calc_snr(
    avg_d: NDArray[np.complex128], std_d: NDArray[np.complex128], axis: int = 0
) -> NDArray[np.complex128]:
    contrast = np.take(avg_d, 1, axis=axis) - np.take(avg_d, 0, axis=axis)  # (*sweep)
    noise2_i = np.sum(std_d.real**2, axis=axis)  # (*sweep)
    noise2_q = np.sum(std_d.imag**2, axis=axis)  # (*sweep)
    noise = np.sqrt(noise2_i * contrast.real**2 + noise2_q * contrast.imag**2) / np.abs(
        contrast
    )

    return contrast / noise


def snr_as_signal(
    raw: Tuple[Sequence[NDArray[np.float64]], Optional[Sequence[NDArray[np.float64]]]],
    axis: int = 0,
) -> NDArray[np.complex128]:
    avg_d, std_d = raw

    assert std_d is not None

    avg_s = avg_d[0][0].dot([1, 1j])  # (ge, *sweep)
    std_s = std_d[0][0].dot([1, 1j])  # (ge, *sweep)

    return calc_snr(avg_s, std_s, axis=axis)
