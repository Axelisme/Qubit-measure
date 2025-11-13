from typing import Tuple

import numpy as np


def calc_mod_energy(energies: np.ndarray, r_f: float) -> np.ndarray:
    return np.mod(energies + r_f / 2, r_f) - r_f / 2


def add_nan_on_discontinue(
    x: np.ndarray, y: np.ndarray, r_f: float
) -> Tuple[np.ndarray, np.ndarray]:
    discontinuities = np.where(np.abs(np.diff(y)) > r_f / 2)[0]
    x_new = np.insert(x, discontinuities + 1, np.nan)
    y_new = np.insert(y, discontinuities + 1, np.nan)

    return x_new, y_new


def calc_collision_mask(
    mod_energies: np.ndarray, target_energy: np.ndarray, threshold: float
) -> np.ndarray:
    distances = np.abs(mod_energies - target_energy[:, None])
    return np.any(distances < threshold, axis=1)
