from __future__ import annotations

import warnings
from types import ModuleType

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from zcu_tools.gui.plot_host import FigureContainer
from zcu_tools.gui.plot_routing import has_current_container

from . import fallback, jupyter, qt


def auto_select_backend() -> ModuleType:
    if has_current_container():
        return qt
    backend = mpl.get_backend().lower()
    if "nbagg" in backend:
        return jupyter
    if any(name in backend for name in ["inline", "agg", "qtagg"]):
        return fallback
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


def instant_plot(fig: Figure) -> None:
    """Show a self-built figure immediately, dispatched by active backend:
    Jupyter ``display``, attach into the active GUI container (Qt), or
    ``fig.show`` (fallback). Use for figures built outside ``make_plot_frame``
    (e.g. custom gridspec layouts) so they render in whichever frontend is live."""
    auto_select_backend().instant_plot(fig)


def refresh_figure(fig: Figure) -> None:
    auto_select_backend().refresh_figure(fig)


def close_figure(fig: Figure) -> None:
    auto_select_backend().close_figure(fig)


def set_figure_container(fig: Figure, container: FigureContainer) -> None:
    """Register a Qt container for fig. No-op when Qt backend is not active."""
    backend = mpl.get_backend().lower()
    if "qtagg" in backend:
        qt.set_figure_container(fig, container)


__all__ = [
    # modules
    "jupyter",
    "fallback",
    "qt",
    # functions
    "make_plot_frame",
    "instant_plot",
    "refresh_figure",
    "close_figure",
    "set_figure_container",
]
