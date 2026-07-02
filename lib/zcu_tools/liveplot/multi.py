from __future__ import annotations

from collections.abc import Hashable
from typing import Generic

from matplotlib.figure import Figure
from typing_extensions import TypeVar

from .backend import refresh_figure
from .base import AbsLivePlot

PlotKey_T = TypeVar("PlotKey_T", bound=Hashable)
Plotter_T = TypeVar("Plotter_T", bound=AbsLivePlot, default=AbsLivePlot)


class MultiLivePlot(AbsLivePlot, Generic[PlotKey_T, Plotter_T]):
    """
    A lifecycle and refresh group for multiple live plotters.
    """

    def __init__(
        self,
        fig: Figure,
        plotters: dict[PlotKey_T, Plotter_T],
    ) -> None:
        self.fig = fig
        self.plotters = plotters

    def clear(self) -> None:
        for plotter in self.plotters.values():
            plotter.clear()

    def refresh(self) -> None:
        refresh_figure(self.fig)

    def __enter__(self) -> MultiLivePlot[PlotKey_T, Plotter_T]:
        for plotter in self.plotters.values():
            plotter.__enter__()

        self.fig.tight_layout()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for plotter in self.plotters.values():
            plotter.__exit__(exc_type, exc_value, traceback)

    def get_plotter(self, key: PlotKey_T) -> Plotter_T:
        return self.plotters[key]
