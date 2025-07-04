from typing import Optional, Tuple

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
        num_lines: int = 1,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        disable: bool = False,
    ) -> None:
        segment = Plot1DSegment(xlabel, ylabel, num_lines, title)
        super().__init__([segment], figsize=figsize, disable=disable)

    def update(
        self,
        xs: np.ndarray,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        ax: plt.Axes = self.axs[0]
        segment: Plot1DSegment = self.segments[0]

        if self.disable:
            return

        with self.update_lock:
            segment.update(ax, xs, signals, title)
            if refresh:
                self._refresh_unchecked()
