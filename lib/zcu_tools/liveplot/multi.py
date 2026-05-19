from __future__ import annotations

from matplotlib.figure import Figure
from typing_extensions import Generic, Hashable, TypeVar

from .backend import refresh_figure
from .base import AbsLivePlot

PlotKey_T = TypeVar("PlotKey_T", bound=Hashable)


class MultiLivePlot(AbsLivePlot, Generic[PlotKey_T]):
    """
    A wrapper for multiple live plotters.

    This class need a dispatch function to dispatch the arguments to the plotters.
    The dispatch function should return a dictionary of arguments for each plotter.
    If the argument is None, the corresponding plotter will not be updated.
    """

    def __init__(
        self,
        fig: Figure,
        plotters: dict[PlotKey_T, AbsLivePlot],
    ) -> None:
        self.fig = fig
        self.plotters = plotters

    def clear(self) -> None:
        for plotter in self.plotters.values():
            plotter.clear()

    def update(self, plot_args: dict[PlotKey_T, tuple], refresh: bool = True) -> None:
        for key, args in plot_args.items():
            self.plotters[key].update(*args, refresh=False)
        if refresh:
            self.refresh()

    def refresh(self) -> None:
        refresh_figure(self.fig)

    def __enter__(self) -> MultiLivePlot[PlotKey_T]:
        for plotter in self.plotters.values():
            plotter.__enter__()

        self.fig.tight_layout()

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for plotter in self.plotters.values():
            plotter.__exit__(exc_type, exc_value, traceback)

    def get_plotter(self, key: PlotKey_T) -> AbsLivePlot:
        return self.plotters[key]
