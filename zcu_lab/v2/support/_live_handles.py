"""Typed update inputs for the lab's live plotting tools."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.plotting import LiveUpdate


@dataclass
class LineData:
    xs: NDArray[np.float64]
    signals: NDArray[np.float64]
    title: str | None


@dataclass
class ScatterData:
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    colors: NDArray[np.float64]
    title: str | None


@dataclass
class HeatmapData:
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    signals: NDArray[np.float64]
    title: str | None


def _heatmap_data(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    signals: NDArray[np.float64],
    title: str | None,
) -> HeatmapData:
    if any(np.iscomplexobj(data) for data in (xs, ys, signals)):
        raise ValueError("Heatmap updates require real-valued data")
    x_data, y_data, signal_data = (
        np.asarray(data, dtype=np.float64) for data in (xs, ys, signals)
    )
    if x_data.ndim != 1 or y_data.ndim != 1 or not x_data.size or not y_data.size:
        raise ValueError("Heatmap axes must be non-empty one-dimensional arrays")
    if signal_data.shape != (x_data.size, y_data.size):
        raise ValueError("Heatmap signals shape must match the x and y axes")
    return HeatmapData(x_data, y_data, signal_data, title)


class LinePlot:
    """Live line updates created by live_line; artists remain on the host owner."""

    def __init__(self, updates: LiveUpdate[LineData], num_lines: int) -> None:
        """Bind owner updates and the factory's positive number of line artists."""
        self._updates = updates
        self._num_lines = num_lines

    def update(
        self,
        xs: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        """Replace real 1D xs and signals shaped (num_lines, len(xs)).

        One line also accepts 1D signals. None preserves the title. Invalid
        data raises ValueError before mutation. LiveUpdate copies the data,
        dispatches to the owner and refreshes unless disabled; finish rejects
        further updates with RuntimeError. Host/drawing errors propagate.
        """
        self._updates.ensure_active()
        if np.iscomplexobj(xs) or np.iscomplexobj(signals):
            raise ValueError("Line plot updates require real-valued data")
        x_data = np.asarray(xs, dtype=np.float64)
        y_data = np.asarray(signals, dtype=np.float64)
        if y_data.ndim == 1:
            y_data = y_data[None, :]
        if (
            x_data.ndim != 1
            or y_data.ndim != 2
            or y_data.shape != (self._num_lines, x_data.size)
        ):
            raise ValueError("Line plot data must match the line count and x length")
        self._updates.update(LineData(x_data, y_data, title), refresh=refresh)

    def refresh(self) -> None:
        """Refresh this live figure on its owner; raise RuntimeError after finish."""
        self._updates.refresh()


class ScatterPlot:
    """Owner-updated scalar-colored points created by live_scatter."""

    def __init__(self, updates: LiveUpdate[ScatterData]) -> None:
        """Bind the factory's snapshot/owner update mechanism."""
        self._updates = updates

    def update(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        *,
        colors: NDArray[np.float64],
        title: str | None = None,
        refresh: bool = True,
    ) -> None:
        """Replace nonempty real 1D xs/ys/colors of equal length.

        Colors are scalar coordinates, not RGB values. None preserves the
        title. Invalid data raises ValueError before mutation. LiveUpdate
        copies data, applies on the owner and refreshes unless disabled.
        Finish rejects further updates; host/drawing errors propagate.
        """
        self._updates.ensure_active()
        if any(np.iscomplexobj(data) for data in (xs, ys, colors)):
            raise ValueError("Scatter updates require real-valued data")
        x_data, y_data, color_data = (
            np.asarray(data, dtype=np.float64) for data in (xs, ys, colors)
        )
        if (
            x_data.ndim != 1
            or x_data.size == 0
            or y_data.shape != x_data.shape
            or color_data.shape != x_data.shape
        ):
            raise ValueError("Scatter data must be nonempty vectors of equal length")
        self._updates.update(
            ScatterData(x_data, y_data, color_data, title), refresh=refresh
        )


class HeatmapPlot:
    """Owner-updated heatmap created by live_heatmap."""

    def __init__(self, updates: LiveUpdate[HeatmapData]) -> None:
        """Bind the factory's snapshot/owner update mechanism."""
        self._updates = updates

    def update(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        """Replace real signals shaped (len(xs), len(ys)) on nonempty 1D axes.

        None preserves the title; a title changes only the heatmap panel.
        Invalid data raises ValueError before mutation. LiveUpdate copies data,
        applies on the owner and refreshes unless disabled. Finish rejects
        further updates; host/drawing errors propagate.
        """
        self._updates.ensure_active()
        self._updates.update(_heatmap_data(xs, ys, signals, title), refresh=refresh)


class HeatmapLinePlot(HeatmapPlot):
    """HeatmapPlot updates also replace the scan lines selected by the factory.

    Use live_heatmap_with_line to create this handle. Input, copying, title,
    refresh and lifetime rules are identical to HeatmapPlot.update.
    """
