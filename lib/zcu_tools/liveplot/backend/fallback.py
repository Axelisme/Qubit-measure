from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def make_plot_frame(
    n_row: int, n_col: int, plot_instant: bool = False, **kwargs
) -> tuple[Figure, list[list[Axes]]]:
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", (6 * n_col, 4 * n_row))
    fig, axs = plt.subplots(n_row, n_col, **kwargs)

    if plot_instant:
        fig.show(warn=False)

    return fig, axs


def refresh_figure(fig: Figure) -> None:
    fig.canvas.draw_idle()
    plt.pause(0.001)


def close_figure(fig: Figure) -> None:
    plt.close(fig)
