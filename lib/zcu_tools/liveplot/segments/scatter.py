from typing import Any, Mapping, Optional

import numpy as np
from matplotlib.collections import PathCollection
from numpy.typing import NDArray

from .base import AbsSegment, Axes


class ScatterSegment(AbsSegment):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: Optional[str] = None,
        show_grid: bool = True,
        scatter_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.show_grid = show_grid

        if scatter_kwargs is None:
            scatter_kwargs = {}

        default_scatter_kwargs = {"marker": "."}
        for k, v in default_scatter_kwargs.items():
            scatter_kwargs.setdefault(k, v)

        self.scatter_kwargs = scatter_kwargs
        self.scatter: Optional[PathCollection] = None

    def init_ax(self, ax: Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)
        if self.show_grid:
            ax.grid()

        self.scatter = ax.scatter([], [], **self.scatter_kwargs)
        if "label" in self.scatter_kwargs:
            ax.legend()

    def update(
        self,
        ax: Axes,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        colors: Optional[NDArray[Any]] = None,
        title: Optional[str] = None,
    ) -> None:
        if self.scatter is None:
            raise RuntimeError("Scatter not initialized.")

        offsets = np.c_[xs.flatten(), ys.flatten()]
        self.scatter.set_offsets(offsets)
        if colors is not None:
            self.scatter.set_array(colors.flatten())

        if title is not None:
            ax.set_title(title)

        ax.ignore_existing_data_limits = True
        ax.update_datalim(self.scatter.get_datalim(ax.transData))
        ax.autoscale_view()

    def clear(self, ax: Axes) -> None:
        ax.clear()
        self.scatter = None
