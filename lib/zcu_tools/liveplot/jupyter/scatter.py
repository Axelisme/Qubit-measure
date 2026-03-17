from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing_extensions import Optional, Sequence, Union

from ..base import AbsLivePlotter
from ..segments import ScatterSegment
from .base import JupyterPlotMixin


class LivePlotterScatter(JupyterPlotMixin, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        *,
        segment_kwargs: Optional[dict] = None,
        existed_axes: Optional[list[list[Axes]]] = None,
        auto_close: bool = True,
        disable: bool = False,
    ) -> None:
        if segment_kwargs is None:
            segment_kwargs = {}
        segment = ScatterSegment(xlabel, ylabel, **segment_kwargs)
        super().__init__(
            [[segment]],
            existed_axes=existed_axes,
            auto_close=auto_close,
            disable=disable,
        )

    def update(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        colors: Union[
            Sequence[str],
            Sequence[tuple[float, float, float]],
            Sequence[tuple[float, float, float, float]],
            NDArray[np.float64],
            None,
        ] = None,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        if self.disable:
            return

        ax = self.get_ax()
        segment = self.get_segment()

        with self.update_lock:
            segment.update(ax, xs, ys, colors, title)
            if refresh:
                self._refresh_while_lock()

    def get_ax(self) -> Axes:
        return self.axs[0][0]

    def get_segment(self) -> ScatterSegment:
        return self.segments[0][0]  # type: ignore
