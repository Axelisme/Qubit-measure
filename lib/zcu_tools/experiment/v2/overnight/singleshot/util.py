from numpy.typing import NDArray
import numpy as np


def calc_populations(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    g_pop, e_pop = signals[..., 0], signals[..., 1]
    return np.stack([g_pop, e_pop, 1 - g_pop - e_pop], axis=-1).real
