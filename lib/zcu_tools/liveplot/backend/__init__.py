from __future__ import annotations

import os
import warnings
from types import ModuleType

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import fallback, jupyter, pyside6


def auto_select_backend() -> ModuleType:
    backend = mpl.get_backend().lower()
    if "nbagg" in backend:
        return jupyter
    if "qtagg" in backend:
        return pyside6
    if "inline" in backend:
        return fallback
    else:
        warnings.warn(
            f"Auto-selected backend for matplotlib is '{backend}', which may not be fully supported."
        )
        return fallback


def make_plot_frame(
    n_row: int, n_col: int, plot_instant: bool = False, **kwargs
) -> tuple[Figure, list[list[Axes]]]:

    return auto_select_backend().make_plot_frame(
        n_row, n_col, plot_instant=plot_instant, **kwargs
    )


def refresh_figure(fig: Figure) -> None:
    auto_select_backend().refresh_figure(fig)


def close_figure(fig: Figure) -> None:
    auto_select_backend().close_figure(fig)


__all__ = [
    # module
    "jupyter",
    "fallback",
    "pyside6",
    # functions
    "make_plot_frame",
    "refresh_figure",
    "close_figure",
]
