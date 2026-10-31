"""Caller-thread artist tools, independent of presentation and plot lifetime."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from ._live_format import (
    HeatmapFormat,
    LineFormat,
    NonUniformHeatmapFormat,
    ScatterFormat,
)
from ._live_handles import heatmap_data, line_data, scatter_data


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


class AttachedHeatmap:
    """Heatmap attached to caller-owned axes, updated with set(xs, ys, signals).

    Add only an image, preserving labels and title. Construction and set run
    on the caller thread without dispatch, refresh or presentation. Use set
    inside @plots.on_plot_thread after display. No axes-claim check is made,
    including against legacy tools. Authors coordinate shared artists,
    autoscale and color limits. This tool does not bind a Plots lifetime.
    Removing its image or axes makes set raise RuntimeError. Matplotlib errors
    propagate without rollback.
    """

    def __init__(
        self,
        axes: Axes,
        *,
        uniform: bool = True,
        clim: tuple[float, float] | None = None,
    ) -> None:
        """Add a nearest-sample image to axes with fixed or automatic color limits.

        uniform=True spans the endpoints evenly; False supports monotonic,
        uneven centers in either scan direction. clim fixes (min, max);
        None rescales to each update's finite data range.
        """
        self._axes = axes
        vmin, vmax = clim if clim is not None else (None, None)
        self._format = (
            HeatmapFormat("", "", vmin=vmin, vmax=vmax)
            if uniform
            else NonUniformHeatmapFormat("", "", vmin=vmin, vmax=vmax)
        )
        self._format.attach_ax(axes)

    def set(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        signals: NDArray[np.float64],
    ) -> tuple[AxesImage, ...]:
        """Replace real signals shaped (len(xs), len(ys)), returning the image.

        xs and ys must be nonempty real 1D arrays. NaN marks unmeasured samples.
        Invalid dimensions, shape or complex input raises ValueError before
        mutation. Update spatial extent and color limits without changing
        labels or title, refreshing or dispatching.
        """
        data = heatmap_data(xs, ys, signals, None)
        self._format.update(self._axes, data.xs, data.ys, data.signals)
        image = self._format.im
        if image is None:
            raise RuntimeError("Image not initialized")
        return (image,)


class AttachedScatter:
    """Scalar-colored points on caller-owned axes, updated with set.

    Add only artists, preserving labels, title and grid settings. Construction
    and set run on the caller thread without dispatch, refresh or presentation.
    Use set inside @plots.on_plot_thread after display. No axes-claim check is
    made, including against legacy tools. Authors coordinate shared artists,
    autoscale and color limits. This tool does not bind a Plots lifetime.
    Removing its artists or axes makes set raise RuntimeError. Matplotlib
    errors propagate without rollback.
    """

    def __init__(self, axes: Axes) -> None:
        """Attach an empty scalar-colored point collection to axes."""
        self._axes = axes
        self._format = ScatterFormat("", "")
        self._format.attach_ax(axes)

    def set(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        *,
        colors: NDArray[np.float64],
    ) -> tuple[PathCollection, ...]:
        """Replace nonempty real 1D xs, ys and scalar colors of equal length.

        NaN marks unmeasured samples. Autoscale spatial and color limits to the
        new data and return the changed collection. Invalid dimensions, shape
        or complex input raises ValueError before mutation. Do not change
        labels or title, dispatch or refresh.
        """
        data = scatter_data(xs, ys, colors)
        self._format.update(self._axes, data.xs, data.ys, colors=data.colors)
        scatter = self._format.scatter
        if scatter is None:
            raise RuntimeError("Scatter not initialized")
        return (scatter,)
