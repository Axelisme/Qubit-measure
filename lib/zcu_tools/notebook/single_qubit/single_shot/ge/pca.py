import numpy as np

from zcu_tools.notebook.single_qubit.process import rotate2real

from .base import fitting_ge_and_plot


def get_rotate_angle(
    Ig: np.ndarray, Qg: np.ndarray, Ie: np.ndarray, Qe: np.ndarray
) -> dict:
    signals = np.concatenate([Ig + 1j * Qg, Ie + 1j * Qe])
    _, angle = rotate2real(signals, ret_angle=True)
    return {"theta": angle}


def fit_ge_by_pca(signals: np.ndarray) -> tuple:
    return fitting_ge_and_plot(signals, get_rotate_angle)
