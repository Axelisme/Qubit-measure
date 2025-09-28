from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..base import AbsLivePlotter
from ..segments import HistogramSegment
from .base import JupyterPlotMixin


class LivePlotterHistogram(JupyterPlotMixin, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        bins: int = 50,
        title: Optional[str] = None,
        disable: bool = False,
    ) -> None:
        segment = HistogramSegment(xlabel, ylabel, title, bins)
        super().__init__([[segment]], disable=disable)

    def update(
        self,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        ax = self.axs[0][0]
        segment = self.segments[0][0]
        assert isinstance(ax, plt.Axes)
        assert isinstance(segment, HistogramSegment)

        if self.disable:
            return

        with self.update_lock:
            segment.update(ax, signals, title)
            if refresh:
                self._refresh_while_lock()
