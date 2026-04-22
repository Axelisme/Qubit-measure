from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing_extensions import Optional, Self

from ..backend import close_figure, make_plot_frame, refresh_figure
from ..base import AbsLivePlot


class AbsSegment(ABC):
    @abstractmethod
    def init_ax(self, ax: Axes) -> None:
        """Initialize the segment with a matplotlib Axes object."""
        pass

    @abstractmethod
    def update(self, ax: Axes, *args, **kwargs) -> None:
        """Update the segment with new data."""
        pass

    @abstractmethod
    def clear(self, ax: Axes) -> None:
        """Clear the segment from the Axes."""
        pass


class BaseSegmentLivePlot(AbsLivePlot):
    def __init__(
        self,
        segments: list[list[AbsSegment]],
        existed_axes: Optional[list[list[Axes]]] = None,
        auto_close: bool = True,
        disable: bool = False,
    ) -> None:
        if len(list(chain.from_iterable(segments))) == 0:
            raise ValueError("At least one segment is required.")
        n_row = len(segments)
        n_col = len(segments[0])

        # validate check
        for s_row in segments:
            if len(s_row) != n_col:
                raise ValueError(
                    "Number of segments in each row must match number of columns."
                )

        self.segments = segments
        self.disable = disable
        self.auto_close = auto_close

        if disable:
            return  # early return

        if existed_axes is not None:
            # validate check
            valid = len(existed_axes) == n_row
            for a_row in existed_axes:
                if len(a_row) != n_col:
                    valid = False
            if not valid:
                raise ValueError(
                    "The shape of provided axes must match the shape of segments."
                )

            # if provided axes and display handle, use them
            self.fig = None
            self.axs = existed_axes
        else:
            # if not provided axes, create figure and display handle
            self.fig, self.axs = make_plot_frame(n_row, n_col, plot_instant=True)

    def clear(self) -> None:
        if self.disable:
            return

        for ax_row, seg_row in zip(self.axs, self.segments):
            for ax, segment in zip(ax_row, seg_row):
                segment.clear(ax)
        self.refresh()

    def refresh(self) -> None:
        if self.disable:
            return

        if self.fig is None:
            raise RuntimeError(
                "Try to refresh a plotter, but it have no own figure, this should happen when the figure is hosted outside of the plotter, you need to refresh the figure manually."
            )

        refresh_figure(self.fig)

    def __enter__(self: Self) -> Self:
        if self.disable:
            return self

        for ax_row, seg_row in zip(self.axs, self.segments):
            for ax, segment in zip(ax_row, seg_row):
                segment.init_ax(ax)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.disable:
            return

        if self.auto_close and self.fig is not None:
            close_figure(self.fig)
