"""Synthetic caller drawing callbacks for frontend and workflow contracts.

These traces exercise figure ownership and presentation, not lab formatting.
"""

from collections.abc import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.plotting import LiveUpdate
from zcu_tools.plotting.plots import PlotHost, Plots

TraceData = tuple[NDArray[np.float64], NDArray[np.float64]]


def create_trace(
    plots: Plots,
    name: str,
    xlabel: str = "",
    ylabel: str = "",
    *,
    axes: Axes | None = None,
    title: str | None = None,
) -> LiveUpdate[TraceData]:
    """Create a single synthetic trace through the public owner callback seam."""

    def initialize(
        host: PlotHost,
        figure: Figure,
        resolved: tuple[Axes, ...],
        active: Callable[[], None],
    ) -> LiveUpdate[TraceData]:
        (ax,) = resolved
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        (line,) = ax.plot([], [])

        def apply(data: TraceData) -> None:
            line.set_data(*data)
            ax.relim()
            ax.autoscale_view()

        return LiveUpdate(host, figure, apply, active)

    return plots.create_live(name, initialize, axes=None if axes is None else (axes,))
