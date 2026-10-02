from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zcu_tools.plotting.plots import Plots

from .base import GE_FitResult, fitting_ge_and_plot


def fit_ge_manual(
    signals: NDArray[np.complex128], angle: float, *, plots: Plots, **kwargs
) -> tuple[float, NDArray[np.float64], GE_FitResult]:
    return fitting_ge_and_plot(
        signals, lambda *_: {"theta": np.pi * angle / 180}, plots=plots, **kwargs
    )
