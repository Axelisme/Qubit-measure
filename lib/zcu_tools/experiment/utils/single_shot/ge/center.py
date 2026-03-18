from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from numpy import float64
from numpy.typing import NDArray

from .base import GE_FitResult, fitting_ge_and_plot


def get_rotate_angle(
    Ig: NDArray[np.float64],
    Qg: NDArray[np.float64],
    Ie: NDArray[np.float64],
    Qe: NDArray[np.float64],
) -> dict[str, Any]:
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    theta = -np.arctan2((ye - yg), (xe - xg))
    return {"theta": theta}


def fit_ge_by_center(
    signals: NDArray[np.complex128], **kwargs
) -> tuple[float, NDArray[float64], GE_FitResult, Figure]:
    return fitting_ge_and_plot(signals, get_rotate_angle, **kwargs)
