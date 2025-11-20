from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from .base import AbsSegment, Axes


class Plot1DSegment(AbsSegment):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: Optional[str] = None,
        num_lines: int = 1,
        show_grid: bool = True,
        line_kwargs: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    ) -> None:
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.num_line = num_lines
        self.title = title
        self.show_grid = show_grid

        if line_kwargs is None:
            line_kwargs = [None] * self.num_line

        assert len(line_kwargs) == self.num_line, (
            f"Expected {self.num_line} line_kwargs, got {len(line_kwargs)}."
        )

        default_line_kwargs = {"marker": ".", "linestyle": "-", "markersize": 5}
        self.line_kwargs: List[Mapping[str, Any]] = []
        for kwargs_i in line_kwargs:
            if kwargs_i is None:
                kwargs_i = dict()

            assert isinstance(kwargs_i, dict)
            for k, v in default_line_kwargs.items():
                kwargs_i.setdefault(k, v)
            self.line_kwargs.append(kwargs_i)

        self.lines: Optional[List[Line2D]] = None

    def init_ax(self, ax: Axes) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)
        if self.show_grid:
            ax.grid()

        lines = []
        for kwargs in self.line_kwargs:
            assert kwargs is not None
            (line,) = ax.plot([], [], **kwargs)
            lines.append(line)
        self.lines = lines

        if any("label" in (kwargs or {}) for kwargs in self.line_kwargs):
            ax.legend()

    def update(
        self,
        ax: Axes,
        xs: NDArray[np.float64],
        signals: NDArray[np.float64],
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

    def clear(self, ax: Axes) -> None:
        ax.clear()
        self.lines = None
