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
        *,
        segment_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if segment_kwargs is None:
            segment_kwargs = {}
        segment = HistogramSegment(xlabel, ylabel, **segment_kwargs)
        super().__init__([[segment]], **kwargs)

    def update(
        self,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        if self.disable:
            return

        ax = self.axs[0][0]
        segment = self.segments[0][0]
        assert isinstance(ax, plt.Axes)
        assert isinstance(segment, HistogramSegment)

        with self.update_lock:
            segment.update(ax, signals, title)
            if refresh:
                self._refresh_while_lock()
