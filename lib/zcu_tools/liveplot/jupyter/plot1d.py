from typing import Optional, List

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..base import AbsLivePlotter
from ..segments import Plot1DSegment
from .base import JupyterPlotMixin


class LivePlotter1D(JupyterPlotMixin, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        *,
        segment_kwargs: Optional[dict] = None,
        existed_axes: Optional[List[List[Axes]]] = None,
        auto_close: bool = True,
        disable: bool = False,
    ) -> None:
        if segment_kwargs is None:
            segment_kwargs = {}
        segment = Plot1DSegment(xlabel, ylabel, **segment_kwargs)
        super().__init__(
            [[segment]],
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

        with self.update_lock:
            segment.update(ax, xs, signals, title)
            if refresh:
                self._refresh_while_lock()

    def get_ax(self) -> Axes:
        return self.axs[0][0]

    def get_segment(self) -> Plot1DSegment:
        return self.segments[0][0]  # type: ignore
