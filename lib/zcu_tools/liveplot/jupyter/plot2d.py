from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..base import AbsLivePlotter
from ..segments import Plot1DSegment, Plot2DSegment, PlotNonUniform2DSegment
from .base import JupyterPlotMixin


class LivePlotter2D(JupyterPlotMixin, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        *,
        uniform: bool = True,
        segment_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if segment_kwargs is None:
            segment_kwargs = {}

        if uniform:
            segment = Plot2DSegment(xlabel, ylabel, **segment_kwargs)
        else:
            segment = PlotNonUniform2DSegment(xlabel, ylabel, **segment_kwargs)

        super().__init__([[segment]], **kwargs)

    def update(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        ax = self.axs[0][0]
        segment = self.segments[0][0]
        assert isinstance(ax, plt.Axes)
        assert isinstance(segment, (PlotNonUniform2DSegment, Plot2DSegment))

        if self.disable:
            return

        with self.update_lock:
            segment.update(ax, xs, ys, signals, title)
            if refresh:
                self._refresh_while_lock()


class LivePlotter2DwithLine(JupyterPlotMixin, AbsLivePlotter):
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        line_axis: Literal[0, 1],
        num_lines: int = 1,
        title: Optional[str] = None,
        uniform: bool = True,
        segment2d_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        if segment2d_kwargs is None:
            segment2d_kwargs = {}

        if uniform:
            segment2d = Plot2DSegment(xlabel, ylabel, title, **segment2d_kwargs)
        else:
            segment2d = PlotNonUniform2DSegment(
                xlabel, ylabel, title, **segment2d_kwargs
            )

        xlabel1d = xlabel if line_axis == 0 else ylabel
        line_kwargs = [
            dict(linestyle="-", markersize=5, alpha=0.3, color="red")
            for _ in range(num_lines)
        ]
        line_kwargs[-1].update(label="current line", marker=".", alpha=1.0, color="C0")

        segment1d = Plot1DSegment(xlabel1d, "", num_lines, line_kwargs=line_kwargs)
        super().__init__([[segment2d, segment1d]], **kwargs)

        self.num_lines = num_lines
        self.line_axis = line_axis

        # set grid to 1d plot
        ax1d = self.axs[0][1]
        assert isinstance(ax1d, plt.Axes)
        ax1d.grid()

    def update(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        signals: np.ndarray,
        title: Optional[str] = None,
        refresh: bool = True,
    ) -> None:
        if self.disable:
            return

        ax2d = self.axs[0][0]
        ax1d = self.axs[0][1]
        segment2d = self.segments[0][0]
        segment1d = self.segments[0][1]
        assert isinstance(ax2d, plt.Axes)
        assert isinstance(ax1d, plt.Axes)
        assert isinstance(segment2d, (PlotNonUniform2DSegment, Plot2DSegment))
        assert isinstance(segment1d, Plot1DSegment)

        # use the last non-nan line as current line
        if np.all(np.isnan(signals)):
            current_line = -1
        else:
            current_line = np.where(~np.isnan(signals))[1 - self.line_axis][-1]

        line_signals = np.full((self.num_lines, signals.shape[self.line_axis]), np.nan)
        for i in range(self.num_lines):
            if current_line - i < 0:
                break

            if self.line_axis == 0:
                line_signals[-i - 1, :] = signals[:, current_line - i].T
            else:
                line_signals[-i - 1, :] = signals[current_line - i, :]

        line_xs = xs if self.line_axis == 0 else ys

        with self.update_lock:
            segment2d.update(ax2d, xs, ys, signals, title)
            segment1d.update(ax1d, line_xs, line_signals)
            if refresh:
                self._refresh_while_lock()
