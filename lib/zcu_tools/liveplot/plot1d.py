from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing_extensions import Any, Optional, cast

from .segments import BaseSegmentLivePlot, Plot1DSegment


class LivePlot1D(BaseSegmentLivePlot):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        *,
        segment_kwargs: Optional[dict[str, Any]] = None,
        existed_axes: Optional[list[list[Axes]]] = None,
        auto_close: bool = True,
        disable: bool = False,
    ) -> None:
        if segment_kwargs is None:
            segment_kwargs = {}

        super().__init__(
            [[Plot1DSegment(xlabel, ylabel, **segment_kwargs)]],
            existed_axes=existed_axes,
            auto_close=auto_close,
            disable=disable,
        )

    def update(
        self,
        xs: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        if self.disable:
            return

        ax = self.get_ax()
        segment = self.get_segment()

        segment.update(ax, xs, signals, title)
        if refresh:
            self.refresh()

    def get_ax(self) -> Axes:
        return self.axs[0][0]

    def get_segment(self) -> Plot1DSegment:
        return cast(Plot1DSegment, self.segments[0][0])
