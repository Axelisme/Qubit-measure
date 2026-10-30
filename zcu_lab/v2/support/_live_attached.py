"""Caller-thread artist tools, independent of presentation and plot lifetime."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from ._live_format import LineFormat
from ._live_handles import line_data


class AttachedLine:
    """Lines attached to caller-owned axes, updated with set(xs, signals).

    Construction adds artists without changing labels, title, ticks or grid.
    Construction and set run only on the caller thread, without copying for
    dispatch, refreshing or presenting. Use set inside @plots.on_plot_thread
    after display. No axes-claim check is made, including against legacy tools;
    authors coordinate shared artists, autoscale and color limits. This tool
    does not bind a Plots lifetime. Removing its artists makes set raise
    RuntimeError. Matplotlib errors propagate without rolling back mutations.
    """

    def __init__(self, axes: Axes, *, num_lines: int = 1) -> None:
        """Add num_lines lines to axes; reject counts below one with ValueError."""
        if num_lines < 1:
            raise ValueError("Line count must be positive")
        self._axes = axes
        self._num_lines = num_lines
        self._format = LineFormat("", "", num_lines=num_lines)
        self._format.attach_ax(axes)

    def set(
        self, xs: NDArray[np.float64], signals: NDArray[np.float64]
    ) -> tuple[Line2D, ...]:
        """Replace real 1D xs and signals shaped (num_lines, len(xs)).

        A single line also accepts 1D signals. NaN marks unmeasured samples.
        Autoscale axes to the new data and return the changed line artists.
        Invalid dimensions, shape or complex input raises ValueError before
        mutation. Labels and title remain unchanged; no refresh or dispatch.
        """
        data = line_data(xs, signals, self._num_lines)
        self._format.update(self._axes, data.xs, data.signals)
        lines = self._format.lines
        if lines is None:
            raise RuntimeError("Lines not initialized")
        return tuple(lines)
