from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..segments import Plot2DSegment
from .base import JupyterLivePlotter


class LivePlotter2D(JupyterLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ):
        segment = Plot2DSegment(xlabel, ylabel, title)
        super().__init__([segment], figsize=figsize)

    def update(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        ax: plt.Axes = self.axs[0]
        segment: Plot2DSegment = self.segments[0]

        with self.update_lock:
            segment.update(ax, xs, ys, signals, title)
            if refresh:
                self._refresh_unchecked()
