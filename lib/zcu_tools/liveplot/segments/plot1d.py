from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .base import AbsSegment


class Plot1DSegment(AbsSegment):
    def __init__(
        self, xlabel: str, ylabel: str, num_line: int = 1, title: Optional[str] = None
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.num_line = num_line
        self.title = title

        self.lines: Optional[List[plt.Line2D]] = None

    def init_ax(self, ax: plt.Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)

        self.lines = [
            ax.plot([], [], marker=".", linestyle="-", markersize=5)[0]
            for _ in range(self.num_line)
        ]

    def update(
        self,
        ax: plt.Axes,
        xs: np.ndarray,
        signals: np.ndarray,
        title: Optional[str] = None,
    ) -> None:
        if self.lines is None:
            raise RuntimeError("Lines not initialized.")

        if signals.ndim == 1:
            signals = signals[None, :]

        for i, line in enumerate(self.lines):
            line.set_data(xs, signals[i, :])

        if title is not None:
            ax.set_title(title)

        ax.relim(visible_only=True)
        ax.autoscale_view()

    def clear(self, ax: plt.Axes) -> None:
        ax.clear()
        self.lines = None
