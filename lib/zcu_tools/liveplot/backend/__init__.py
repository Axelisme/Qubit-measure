from __future__ import annotations

import warnings
from types import ModuleType

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def auto_select_backend() -> ModuleType:
    backend = mpl.get_backend().lower()
    if "nbagg" in backend:
        from . import jupyter
        return jupyter
    if "qtagg" in backend:
        from . import pyside6
        return pyside6
    if "inline" in backend:
        from . import fallback
        return fallback
    else:
        warnings.warn(
            f"Auto-selected backend for matplotlib is '{backend}', which may not be fully supported."
        )
        from . import fallback
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
    # functions
    "make_plot_frame",
    "refresh_figure",
    "close_figure",
]
