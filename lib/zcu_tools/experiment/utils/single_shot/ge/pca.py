from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.utils.process import find_rotate_angle

from .base import GE_FitResult, fitting_ge_and_plot


def get_rotate_angle(
    Ig: NDArray[np.float64],
    Qg: NDArray[np.float64],
    Ie: NDArray[np.float64],
    Qe: NDArray[np.float64],
) -> dict[str, float]:
    signals = np.concatenate([Ig + 1j * Qg, Ie + 1j * Qe])
    angle = find_rotate_angle(signals)
    return {"theta": -angle}


def fit_ge_by_pca(
    signals: NDArray[np.complex128], **kwargs
) -> tuple[float, NDArray[np.float64], GE_FitResult, Figure]:
    return fitting_ge_and_plot(signals, get_rotate_angle, **kwargs)
