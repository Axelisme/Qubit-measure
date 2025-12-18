from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def calc_mod_energy(energies: NDArray[np.float64], r_f: float) -> NDArray[np.float64]:
    return np.mod(energies + r_f / 2, r_f) - r_f / 2


def add_nan_on_discontinue(
    x: NDArray[np.float64], y: NDArray[np.float64], r_f: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    discontinuities = np.where(np.abs(np.diff(y)) > r_f / 2)[0]
    x_new = np.insert(x, discontinuities + 1, np.nan)
    y_new = np.insert(y, discontinuities + 1, np.nan)

    return x_new, y_new


def calc_collision_mask(
    mod_energies: NDArray[np.float64],
    target_energy: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.bool_]:
    distances = np.abs(mod_energies - target_energy[:, None])
    return np.any(distances < threshold, axis=1)
