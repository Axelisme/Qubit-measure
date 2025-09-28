from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..base import AbsLivePlotter
from ..segments import Plot1DSegment
from .base import JupyterPlotMixin


class LivePlotter1D(JupyterPlotMixin, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        disable: bool = False,
        **kwargs,
    ) -> None:
        segment = Plot1DSegment(xlabel, ylabel, **kwargs)
        super().__init__([[segment]], disable=disable)

    def update(
        self,
        xs: np.ndarray,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        ax = self.axs[0][0]
        segment = self.segments[0][0]
        assert isinstance(ax, plt.Axes)
        assert isinstance(segment, Plot1DSegment)

        if self.disable:
            return

        with self.update_lock:
            segment.update(ax, xs, signals, title)
            if refresh:
                self._refresh_while_lock()
