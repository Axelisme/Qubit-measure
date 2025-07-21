from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .base import AbsSegment


class Plot1DSegment(AbsSegment):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        num_lines: int = 1,
        title: Optional[str] = None,
        line_kwargs: Optional[List[Optional[dict]]] = None,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.num_line = num_lines
        self.title = title

        if line_kwargs is None:
            line_kwargs = [None] * num_lines
        assert line_kwargs is not None

        assert len(line_kwargs) == self.num_line, (
            f"Expected {self.num_line} line_kwargs, got {len(self.line_kwargs)}."
        )

        self.lines: Optional[List[plt.Line2D]] = None
        self.line_kwargs = line_kwargs

    def init_ax(self, ax: plt.Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)

        lines = []
        for kwargs in self.line_kwargs:
            if kwargs is None:
                kwargs = {"marker": ".", "linestyle": "-", "markersize": 5}

            (line,) = ax.plot([], [], **kwargs)
            lines.append(line)
        self.lines = lines

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
        elif signals.ndim > 2:
            signals = signals.reshape(-1, signals.shape[-1])

        for i, line in enumerate(self.lines):
            line.set_data(xs, signals[i, :])

        if title is not None:
            ax.set_title(title)

        ax.relim(visible_only=True)
        ax.autoscale_view()

    def clear(self, ax: plt.Axes) -> None:
        ax.clear()
        self.lines = None
