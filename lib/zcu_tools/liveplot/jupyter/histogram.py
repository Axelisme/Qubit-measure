from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..base import AbsLivePlotter
from ..segments import HistogramSegment
from .base import JupyterLivePlotter


class LivePlotterHistogram(JupyterLivePlotter, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        bins: int = 50,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        disable: bool = False,
    ):
        segment = HistogramSegment(xlabel, ylabel, title, bins)
        super().__init__([segment], figsize=figsize, disable=disable)

    def update(
        self,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        ax: plt.Axes = self.axs[0]
        segment: HistogramSegment = self.segments[0]

        if self.disable:
            return

        with self.update_lock:
            segment.update(ax, signals, title)
            if refresh:
                self._refresh_unchecked()

    def __enter__(self) -> "LivePlotterHistogram":
        return super().__enter__()
