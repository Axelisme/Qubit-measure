import numpy as np

from .base import fitting_ge_and_plot


def fit_ge_manual(signals: np.ndarray, angle, plot: bool = True) -> tuple:
    return fitting_ge_and_plot(signals, lambda *_: {"theta": np.pi * angle / 180}, plot)
