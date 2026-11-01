"""Synthetic caller drawing callbacks for frontend and workflow contracts.

These traces exercise figure ownership and presentation, not lab formatting.
"""

from collections.abc import Callable, Iterable

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.plotting.plots import Plots

TraceData = tuple[NDArray[np.float64], NDArray[np.float64]]


def create_trace(
    plots: Plots,
    name: str,
    xlabel: str = "",
    ylabel: str = "",
    *,
    title: str | None = None,
) -> Callable[[TraceData], None]:
    """Create and show a synthetic trace; return its decorated owner update."""
    figure, axes = plots.subplots(num=name)
    axes.set(xlabel=xlabel, ylabel=ylabel)
    if title is not None:
        axes.set_title(title)
    (line,) = axes.plot([], [])

    @plots.on_plot_thread
    def update(data: TraceData) -> Iterable[Artist]:
        line.set_data(*data)
        axes.relim()
        axes.autoscale_view()
        return (line,)

    figure.show()
    return update
