from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import LivePlotBackend


class FallbackBackend(LivePlotBackend):
    """Plain matplotlib rendering (``fig.show`` / ``draw_idle``).

    Used outside notebook and GUI. Under the GUI custom mpl backend this is
    also the default: ``plt.subplots`` is intercepted and attached into the
    container, and ``draw_idle`` is marshalled to the main thread by the GUI
    canvas — so the same plain-matplotlib calls render in the GUI transparently
    (``plt.pause`` is skipped because ``isinteractive()`` is False there).
    """

    def make_plot_frame(
        self, n_row: int, n_col: int, plot_instant: bool = False, **kwargs: Any
    ) -> tuple[Figure, list[list[Axes]]]:
        kwargs.setdefault("squeeze", False)
        kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
        fig, axs_nd = plt.subplots(n_row, n_col, **kwargs)
        # plt.subplots(squeeze=False) returns ndarray; convert to list[list[Axes]]
        # to satisfy the LivePlotBackend contract.
        import numpy as np

        axs: list[list[Axes]] = np.asarray(axs_nd).tolist()
        if plot_instant:
            fig.show(warn=False)
        return fig, axs

    def instant_plot(self, fig: Figure) -> None:
        fig.show(warn=False)

    def refresh_figure(self, fig: Figure) -> None:
        fig.canvas.draw_idle()
        if plt.isinteractive():
            plt.pause(0.001)

    def close_figure(self, fig: Figure) -> None:
        plt.close(fig)
