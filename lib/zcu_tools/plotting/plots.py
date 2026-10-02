"""Explicit operation plotting with typed updates and frontend-owned presentation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

if TYPE_CHECKING:
    from matplotlib.animation import AbstractMovieWriter
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .figures import FigureCollection, NamedFigures
from .liveplot.segments.plot1d import Plot1DSegment
from .liveplot.segments.plot2d import Plot2DSegment, PlotNonUniform2DSegment
from .liveplot.segments.scatter import ScatterSegment

_T = TypeVar("_T")


class PlotHost(Protocol):
    """Execute plotting work on one owner; never infer an ambient destination."""

    def call(self, callback: Callable[[], _T]) -> _T: ...

    def present(self, figure: Figure) -> None: ...

    def refresh(self, figure: Figure, *, final: bool = False) -> None: ...

    def release(self, figure: Figure) -> None: ...


class NonPresentingHost:
    """Keep native artists and savefig usable without rendering or opening a view."""

    def call(self, callback: Callable[[], _T]) -> _T:
        return callback()

    def present(self, figure: Figure) -> None:
        pass

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        pass

    def release(self, figure: Figure) -> None:
        pass


class LinePlot:
    """A typed update handle; active artists belong to the plots host's owner."""

    def __init__(
        self,
        host: PlotHost,
        figure: Figure,
        axes: Axes,
        segment: Plot1DSegment,
        ensure_active: Callable[[], None],
    ) -> None:
        self._host = host
        self._figure = figure
        self._axes = axes
        self._segment = segment
        self._ensure_active = ensure_active

    def update(
        self,
        xs: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        """Copy real data before dispatch; reject invalid shapes before mutation."""
        self._ensure_active()
        if np.iscomplexobj(xs) or np.iscomplexobj(signals):
            raise ValueError("Line plot updates require real-valued data")
        x_data = np.array(xs, dtype=np.float64, copy=True)
        y_data = np.array(signals, dtype=np.float64, copy=True)
        if y_data.ndim == 1:
            y_data = y_data[None, :]
        if (
            x_data.ndim != 1
            or y_data.ndim != 2
            or y_data.shape != (self._segment.num_line, x_data.size)
        ):
            raise ValueError("Line plot data must match the line count and x length")

        def apply() -> None:
            self._ensure_active()
            self._segment.update(self._axes, x_data, y_data, title)
            if refresh:
                self._host.refresh(self._figure)

        self._host.call(apply)

    def refresh(self) -> None:
        self._ensure_active()
        self._host.call(lambda: self._host.refresh(self._figure))


class ScatterPlot:
    """Owner-updated points with one real-valued color coordinate per sample."""

    def __init__(
        self,
        host: PlotHost,
        figure: Figure,
        axes: Axes,
        segment: ScatterSegment,
        ensure_active: Callable[[], None],
    ) -> None:
        self._host = host
        self._figure = figure
        self._axes = axes
        self._segment = segment
        self._ensure_active = ensure_active

    def update(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        *,
        colors: NDArray[np.float64],
        title: str | None = None,
        refresh: bool = True,
    ) -> None:
        """Copy equally sized, nonempty real vectors before owner dispatch."""
        self._ensure_active()
        if any(np.iscomplexobj(data) for data in (xs, ys, colors)):
            raise ValueError("Scatter updates require real-valued data")
        x_data, y_data, color_data = (
            np.array(data, dtype=np.float64, copy=True) for data in (xs, ys, colors)
        )
        if (
            x_data.ndim != 1
            or x_data.size == 0
            or y_data.shape != x_data.shape
            or color_data.shape != x_data.shape
        ):
            raise ValueError("Scatter data must be nonempty vectors of equal length")

        def apply() -> None:
            self._ensure_active()
            self._segment.update(
                self._axes, x_data, y_data, colors=color_data, title=title
            )
            if refresh:
                self._host.refresh(self._figure)

        self._host.call(apply)


def _heatmap_data(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    signals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if any(np.iscomplexobj(data) for data in (xs, ys, signals)):
        raise ValueError("Heatmap updates require real-valued data")
    x_data = np.array(xs, dtype=np.float64, copy=True)
    y_data = np.array(ys, dtype=np.float64, copy=True)
    signal_data = np.array(signals, dtype=np.float64, copy=True)
    if x_data.ndim != 1 or y_data.ndim != 1 or not x_data.size or not y_data.size:
        raise ValueError("Heatmap axes must be non-empty one-dimensional arrays")
    if signal_data.shape != (x_data.size, y_data.size):
        raise ValueError("Heatmap signals shape must match the x and y axes")

    return x_data, y_data, signal_data


class HeatmapPlot:
    """Update one named heatmap through its presentation owner."""

    def __init__(
        self,
        host: PlotHost,
        figure: Figure,
        axes: Axes,
        heatmap: Plot2DSegment | PlotNonUniform2DSegment,
        ensure_active: Callable[[], None],
    ) -> None:
        self._host = host
        self._figure = figure
        self._axes = axes
        self._heatmap = heatmap
        self._ensure_active = ensure_active
        self._marker: PathCollection | None = None

    def mark_point(self, x: float, y: float) -> None:
        """Set the red best-point overlay; NaN coordinates hide it."""
        self._ensure_active()
        point = np.array([[x, y]], dtype=np.float64)

        def apply() -> None:
            self._ensure_active()
            if self._marker is None:
                self._marker = self._axes.scatter(
                    [], [], color="red", label="Best Point", zorder=3
                )
            self._marker.set_offsets(point)

        self._host.call(apply)

    def update(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        self._ensure_active()
        x_data, y_data, signal_data = _heatmap_data(xs, ys, signals)

        def apply() -> None:
            self._ensure_active()
            self._heatmap.update(self._axes, x_data, y_data, signal_data, title)
            if refresh:
                self._host.refresh(self._figure)

        self._host.call(apply)


class HeatmapLinePlot:
    """Update a named 2D heatmap and its most recent scan lines on one owner."""

    def __init__(  # noqa: PLR0913 - retain the owner and both segment/axis pairs
        self,
        host: PlotHost,
        figure: Figure,
        heatmap_axes: Axes,
        line_axes: Axes,
        heatmap: Plot2DSegment | PlotNonUniform2DSegment,
        lines: Plot1DSegment,
        line_axis: Literal[0, 1],
        ensure_active: Callable[[], None],
    ) -> None:
        self._host = host
        self._figure = figure
        self._heatmap_axes = heatmap_axes
        self._line_axes = line_axes
        self._heatmap = heatmap
        self._lines = lines
        self._line_axis = line_axis
        self._ensure_active = ensure_active
        self._marker: Line2D | None = None

    def mark_line(self, x: float) -> None:
        """Set a red dashed reference on the scan-line axes; NaN hides it."""
        self._ensure_active()
        position = float(x)

        def apply() -> None:
            self._ensure_active()
            if self._marker is None:
                self._marker = self._line_axes.axvline(
                    np.nan, color="red", linestyle="--"
                )
            self._marker.set_xdata([position])

        self._host.call(apply)

    def update(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        signals: NDArray[np.float64],
        title: str | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        """Validate and copy producer data before dispatching artist mutation."""
        self._ensure_active()
        x_data, y_data, signal_data = _heatmap_data(xs, ys, signals)

        line_xs = x_data if self._line_axis == 0 else y_data
        # The final non-empty row/column is current; earlier lines retain scan order.
        valid = ~np.all(np.isnan(signal_data), axis=self._line_axis)
        current = int(np.flatnonzero(valid)[-1]) if np.any(valid) else -1
        line_data = np.full((self._lines.num_line, line_xs.size), np.nan)
        for offset in range(self._lines.num_line):
            index = current - offset
            if index < 0:
                break
            line_data[-offset - 1] = (
                signal_data[:, index] if self._line_axis == 0 else signal_data[index, :]
            )

        def apply() -> None:
            self._ensure_active()
            self._heatmap.update(self._heatmap_axes, x_data, y_data, signal_data, title)
            self._lines.update(self._line_axes, line_xs, line_data)
            if refresh:
                self._host.refresh(self._figure)

        self._host.call(apply)


class MovieRecording:
    """Capture the live figure on its host owner; finish does not release it."""

    def __init__(
        self,
        host: PlotHost,
        writer: AbstractMovieWriter,
        ensure_active: Callable[[], None],
    ) -> None:
        self._host = host
        self._writer = writer
        self._ensure_active = ensure_active
        self._closed = False

    def grab_frame(self) -> None:
        def capture() -> None:
            self._ensure_active()
            if self._closed:
                raise RuntimeError("Movie recording has finished")
            self._writer.grab_frame()

        self._host.call(capture)

    def finish(self) -> None:
        def close() -> None:
            if not self._closed:
                self._closed = True
                self._writer.finish()

        self._host.call(close)


class Plots(FigureCollection):
    """One producer's plotting lifetime, shared by frontend adapters.

    Stop the producer before finish. Host calls are synchronous and preserve
    that producer's order. General figures stay unpresented until finish, while
    live figures present immediately. Finishing fixes membership and stops typed
    updates; it does not declare analysis success. Release presentation separately
    after finishing. Neither finish nor release destroys retained Figure objects.
    """

    def __init__(self, host: PlotHost) -> None:
        super().__init__()
        self._host = host
        self._live: list[Figure] = []
        self._live_axes: set[Axes] = set()
        self._recordings: dict[Figure, MovieRecording] = {}
        self._finished = False
        self._finished_figures: NamedFigures | None = None
        self._released: set[Figure] = set()

    def _ensure_active(self) -> None:
        if self._finished:
            raise RuntimeError("Plot operation has finished")

    def _resolve_axes(
        self, name: str, axes: tuple[Axes, ...] | None, count: int
    ) -> tuple[Figure, tuple[Axes, ...]]:
        if axes is None:
            figure, _ = self.subplots(name, ncols=count, squeeze=False)
            resolved = tuple(figure.axes)
        else:
            figure = self[name]
            resolved = axes
        if len(resolved) != count or len(set(resolved)) != count:
            raise ValueError("Live plot requires distinct axes of the expected count")
        if any(ax.figure is not figure or ax not in figure.axes for ax in resolved):
            raise ValueError("Live axes must belong to the named figure")
        if any(ax in self._live_axes for ax in resolved):
            raise ValueError("Axes already belong to a live plot")
        self._live_axes.update(resolved)
        return figure, resolved

    def _present_live(self, figure: Figure) -> None:
        if figure not in self._live:
            self._host.present(figure)
            self._live.append(figure)

    def refresh(self, name: str) -> None:
        """Refresh a complete live frame after a batch of typed updates."""
        self._ensure_active()
        figure = self[name]
        if figure not in self._live:
            raise ValueError("Figure is not live")

        def refresh_frame() -> None:
            self._ensure_active()
            self._host.refresh(figure)

        self._host.call(refresh_frame)

    def record_animation(self, name: str, path: str | Path) -> MovieRecording:
        """Record a live figure with FFmpeg, without changing its presentation.

        The producer calls grab_frame after each complete update. Finish the
        recorder when acquisition ends; Plots.finish also closes open recorders.
        Writer errors propagate and do not declare the operation successful.
        """
        from matplotlib.animation import FFMpegWriter

        self._ensure_active()
        figure = self[name]
        if figure not in self._live:
            raise ValueError("Figure is not live")
        if figure in self._recordings:
            raise ValueError("Figure already has a movie recording")
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("FFmpeg is required to record animations")

        def start() -> MovieRecording:
            self._ensure_active()
            writer = FFMpegWriter(fps=30)
            writer.setup(figure, str(path), dpi=200)
            recording = MovieRecording(self._host, writer, self._ensure_active)
            self._recordings[figure] = recording
            return recording

        return self._host.call(start)

    def liveplot_1d(  # noqa: PLR0913 - explicit style and optional native axes
        self,
        name: str,
        xlabel: str,
        ylabel: str,
        *,
        title: str | None = None,
        num_lines: int = 1,
        configure_axes: Callable[[Axes], None] | None = None,
        axes: Axes | None = None,
    ) -> LinePlot:
        """Configure native artists once on the host owner before presenting.

        The callback must not retain active axes for later worker-side mutation.
        Configuration errors propagate without presenting the incomplete figure.
        """
        self._ensure_active()
        if num_lines < 1:
            raise ValueError("Line count must be positive")

        def create() -> LinePlot:
            self._ensure_active()
            figure, (ax,) = self._resolve_axes(
                name, None if axes is None else (axes,), 1
            )
            segment = Plot1DSegment(xlabel, ylabel, title=title, num_lines=num_lines)
            segment.init_ax(ax)
            if configure_axes is not None:
                configure_axes(ax)
            viewer = LinePlot(self._host, figure, ax, segment, self._ensure_active)
            self._present_live(figure)
            return viewer

        return self._host.call(create)

    def liveplot_scatter(
        self,
        name: str,
        xlabel: str,
        ylabel: str,
        *,
        title: str | None = None,
    ) -> ScatterPlot:
        """Present scalar-colored points, with artist updates on the host owner."""
        self._ensure_active()

        def create() -> ScatterPlot:
            self._ensure_active()
            figure, axes = self.subplots(name)
            segment = ScatterSegment(xlabel, ylabel, title=title)
            segment.init_ax(axes)
            viewer = ScatterPlot(self._host, figure, axes, segment, self._ensure_active)
            self._host.present(figure)
            self._live.append(figure)
            return viewer

        return self._host.call(create)

    def liveplot_2d(  # noqa: PLR0913 - explicit heatmap choices and optional native axes
        self,
        name: str,
        xlabel: str,
        ylabel: str,
        *,
        title: str | None = None,
        uniform: bool = True,
        clim: tuple[float, float] | None = None,
        axes: Axes | None = None,
    ) -> HeatmapPlot:
        """Present a named heatmap without adding scan-line axes."""
        self._ensure_active()

        def create() -> HeatmapPlot:
            self._ensure_active()
            figure, (ax,) = self._resolve_axes(
                name, None if axes is None else (axes,), 1
            )
            vmin, vmax = clim if clim is not None else (None, None)
            heatmap = (
                Plot2DSegment(xlabel, ylabel, title, vmin=vmin, vmax=vmax)
                if uniform
                else PlotNonUniform2DSegment(
                    xlabel, ylabel, title, vmin=vmin, vmax=vmax
                )
            )
            heatmap.init_ax(ax)
            viewer = HeatmapPlot(self._host, figure, ax, heatmap, self._ensure_active)
            self._present_live(figure)
            return viewer

        return self._host.call(create)

    def liveplot_2d_with_line(  # noqa: PLR0913 - keep scan/axis choices explicit
        self,
        name: str,
        xlabel: str,
        ylabel: str,
        *,
        line_axis: Literal[0, 1] = 1,
        num_lines: int = 1,
        title: str | None = None,
        uniform: bool = True,
        axes: tuple[Axes, Axes] | None = None,
    ) -> HeatmapLinePlot:
        """Present a named, owner-updated 2D heatmap with recent scan lines."""
        self._ensure_active()
        if num_lines < 1:
            raise ValueError("Line count must be positive")
        if line_axis not in (0, 1):
            raise ValueError("line_axis must be 0 or 1")

        def create() -> HeatmapLinePlot:
            self._ensure_active()
            figure, (heatmap_axes, line_axes) = self._resolve_axes(name, axes, 2)
            heatmap = (
                Plot2DSegment(xlabel, ylabel, title)
                if uniform
                else PlotNonUniform2DSegment(xlabel, ylabel, title)
            )
            line_kwargs = [
                {"marker": "None", "alpha": 0.3, "color": "red"}
                for _ in range(num_lines)
            ]
            line_kwargs[-1].update(
                {"label": "current line", "marker": ".", "alpha": 1.0, "color": "C0"}
            )
            lines = Plot1DSegment(
                xlabel if line_axis == 0 else ylabel,
                "",
                title=title,
                num_lines=num_lines,
                line_kwargs=line_kwargs,
            )
            heatmap.init_ax(heatmap_axes)
            lines.init_ax(line_axes)
            viewer = HeatmapLinePlot(
                self._host,
                figure,
                heatmap_axes,
                line_axes,
                heatmap,
                lines,
                line_axis,
                self._ensure_active,
            )
            self._present_live(figure)
            return viewer

        return self._host.call(create)

    def finish(self, *, present: bool = True) -> NamedFigures:
        if self._finished_figures is not None:
            return self._finished_figures
        self._finished = True
        figures = NamedFigures(self)
        self._finished_figures = figures

        def complete() -> None:
            errors: list[Exception] = []
            for recording in self._recordings.values():
                try:
                    recording.finish()
                except Exception as error:  # noqa: BLE001 - finish all recordings
                    errors.append(error)
            if errors:
                raise ExceptionGroup("Failed to finish movie recordings", errors)
            for figure in self._live:
                self._host.refresh(figure, final=True)
            if present:
                for figure in self.values():
                    if figure not in self._live:
                        self._host.present(figure)

        self._host.call(complete)
        return figures

    def release(self) -> None:
        """Release every presentation, reporting failures after trying all figures."""
        if not self._finished:
            raise RuntimeError(
                "Finish the plot operation before releasing presentation"
            )

        def cleanup() -> None:
            errors: list[Exception] = []
            for figure in self.values():
                if figure in self._released:
                    continue
                try:
                    self._host.release(figure)
                except Exception as error:  # noqa: BLE001 - report all cleanup failures together
                    errors.append(error)
                else:
                    self._released.add(figure)
            if errors:
                raise ExceptionGroup("Failed to release plot presentation", errors)

        self._host.call(cleanup)
