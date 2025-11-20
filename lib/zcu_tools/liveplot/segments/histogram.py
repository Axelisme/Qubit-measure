from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import AbsSegment, Axes


class HistogramSegment(AbsSegment):
    def __init__(
        self, xlabel: str, ylabel: str, title: Optional[str] = None, bins: int = 50
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.bins = bins

        self.hist = None

    def init_ax(self, ax: Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)

    def update(
        self,
        ax: Axes,
        signals: NDArray[np.float64],
        title: Optional[str] = None,
    ) -> None:
        if title is not None:
            self.title = title

        # redraw the plot because the histogram is not updatable
        ax.clear()

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)

        # Create histogram from signals data
        ax.hist(signals, bins=self.bins, alpha=0.7)

    def clear(self, ax: Axes) -> None:
        ax.clear()
