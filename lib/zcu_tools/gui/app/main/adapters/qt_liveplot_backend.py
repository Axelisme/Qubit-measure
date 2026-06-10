"""Qt liveplot backend — the GUI's implementation of ``LivePlotBackend``.

A Driven Adapter: the GUI registers this with liveplot (``set_liveplot_backend``
in the run worker), so the dependency runs gui → liveplot. liveplot never
imports or detects the GUI; it just drives whatever backend is registered.

All four operations route through the GUI plotting substrate:
- ``make_plot_frame`` builds with ``plt.subplots`` — the GUI custom mpl backend
  (``GuiFigureManager``) intercepts it and attaches the figure into the active
  ``FigureContainer`` on the main thread. Same render path as a bare
  ``plt.subplots()`` and as analysis figures.
- ``refresh_figure`` marshals the repaint to the main thread (worker-safe).
- ``close_figure`` is a no-op: figure lifetime is owned by the container and
  cleared on the next run start.
- ``instant_plot`` is a no-op: a pyplot figure is already attached at creation.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from zcu_tools.gui.plotting import refresh_figure_in_main_thread
from zcu_tools.liveplot.backend.base import LivePlotBackend


class QtLivePlotBackend(LivePlotBackend):
    def make_plot_frame(
        self, n_row: int, n_col: int, plot_instant: bool = False, **kwargs: Any
    ) -> tuple[Figure, list[list[Axes]]]:
        del plot_instant  # figure is attached at creation; nothing to show
        kwargs.setdefault("squeeze", False)
        kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
        return plt.subplots(n_row, n_col, **kwargs)

    def instant_plot(self, fig: Figure) -> None:
        del fig  # already attached to the FigureContainer at creation time

    def refresh_figure(self, fig: Figure) -> None:
        refresh_figure_in_main_thread(fig)

    def close_figure(self, fig: Figure) -> None:
        del fig  # Qt host manages figure lifetime; cleared on next run start
