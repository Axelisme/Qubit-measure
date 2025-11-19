from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .base import AbsSegment


class Plot1DSegment(AbsSegment):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: Optional[str] = None,
        num_lines: int = 1,
        show_grid: bool = True,
        line_kwargs: Optional[List[Optional[dict]]] = None,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.num_line = num_lines
        self.title = title
        self.show_grid = show_grid

        if line_kwargs is None:
            line_kwargs = [None] * self.num_line
        assert line_kwargs is not None

        assert len(line_kwargs) == self.num_line, (
            f"Expected {self.num_line} line_kwargs, got {len(line_kwargs)}."
        )

        default_line_kwargs = {"marker": ".", "linestyle": "-", "markersize": 5}
        for i in range(self.num_line):
            if line_kwargs[i] is None:
                line_kwargs[i] = dict()
            for k, v in default_line_kwargs.items():
                line_kwargs[i].setdefault(k, v)

        self.lines: Optional[List[plt.Line2D]] = None
        self.line_kwargs = line_kwargs

    def init_ax(self, ax: plt.Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)
        if self.show_grid:
            ax.grid()

        lines = []
        for kwargs in self.line_kwargs:
            (line,) = ax.plot([], [], **kwargs)
            lines.append(line)
        self.lines = lines

        if any("label" in (kwargs or {}) for kwargs in self.line_kwargs):
            ax.legend()

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
            line.set_data(xs, signals[i, :].astype(np.float64))

        if title is not None:
            ax.set_title(title)

        ax.relim(visible_only=True)
        ax.autoscale_view()

    def clear(self, ax: plt.Axes) -> None:
        ax.clear()
        self.lines = None
