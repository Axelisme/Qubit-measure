import numpy as np

from zcu_tools.utils.process import find_rotate_angle

from .base import fitting_ge_and_plot


def get_rotate_angle(
    Ig: np.ndarray, Qg: np.ndarray, Ie: np.ndarray, Qe: np.ndarray
) -> dict:
    signals = np.concatenate([Ig + 1j * Qg, Ie + 1j * Qe])
    angle = find_rotate_angle(signals)
    return {"theta": -angle}


def fit_ge_by_pca(signals: np.ndarray, **kwargs) -> tuple:
    return fitting_ge_and_plot(signals, get_rotate_angle, **kwargs)
