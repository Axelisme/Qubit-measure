from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..base import AbsLivePlotter
from ..segments import Plot1DSegment, Plot2DSegment
from .base import JupyterPlotMixin


class LivePlotter2D(JupyterPlotMixin, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: Optional[str] = None,
        flip: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        disable: bool = False,
    ) -> None:
        segment = Plot2DSegment(xlabel, ylabel, title, flip=flip)
        super().__init__([segment], figsize=figsize, disable=disable)

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

        if self.disable:
            return

        with self.update_lock:
            segment.update(ax, xs, ys, signals, title)
            if refresh:
                self._refresh_unchecked()


class LivePlotter2DwithLine(JupyterPlotMixin, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        line_axis: Literal[0, 1],
        num_lines: int = 1,
        title: Optional[str] = None,
        flip: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        disable: bool = False,
    ) -> None:
        segment2d = Plot2DSegment(xlabel, ylabel, title, flip=flip)
        xlbael1d = xlabel if line_axis == 0 else ylabel
        segment1d = Plot1DSegment(xlbael1d, "", num_lines)
        super().__init__([segment2d, segment1d], figsize=figsize, disable=disable)

        self.num_lines = num_lines
        self.line_axis = line_axis

    def update(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        ax2d: plt.Axes = self.axs[0]
        ax1d: plt.Axes = self.axs[1]
        segment2d: Plot2DSegment = self.segments[0]
        segment1d: Plot1DSegment = self.segments[1]

        if self.disable:
            return

        # use the last non-nan line as current line
        current_line = np.where(~np.isnan(signals))[1 - self.line_axis][-1]
        line_start = max(0, current_line - self.num_lines + 1)
        if self.line_axis == 0:
            lines_signals = signals[:, line_start:]
            line_xs = xs
        else:
            lines_signals = signals[line_start:, :]
            line_xs = ys

        with self.update_lock:
            segment2d.update(ax2d, xs, ys, signals, title)
            segment1d.update(ax1d, line_xs, lines_signals)
            if refresh:
                self._refresh_unchecked()
