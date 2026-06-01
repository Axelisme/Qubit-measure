"""Qt liveplot backend backed by the GUI plot routing + host layers."""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from zcu_tools.gui.plot_host import (
    FigureContainer,
    attach_existing_figure_to_container,
    create_figure_in_current_container,
    refresh_figure_in_main_thread,
)


def set_figure_container(fig: Figure, container: FigureContainer) -> None:
    attach_existing_figure_to_container(fig, container)


def instant_plot(fig: Figure) -> None:  # noqa: ARG001
    """No-op under the GUI backend: a pyplot-created figure (``plt.figure`` /
    ``plt.subplots``) is already attached to the active FigureContainer by the
    custom matplotlib backend (``gui.mpl_backend.GuiFigureManager``) at creation
    time. Jupyter needs an explicit ``display`` to render right now; the GUI does
    not, so this dispatch target intentionally does nothing."""


def make_plot_frame(
    n_row: int,
    n_col: int,
    plot_instant: bool = False,  # noqa: ARG001
    **kwargs: Any,
) -> tuple[Figure, list[list[Axes]]]:
    return create_figure_in_current_container(n_row, n_col, **kwargs)


def refresh_figure(fig: Figure) -> None:
    refresh_figure_in_main_thread(fig)


def close_figure(fig: Figure) -> None:
    del fig  # Qt host manages figure lifetime; cleared on next run start
