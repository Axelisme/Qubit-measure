"""Explicit operation plotting with typed updates and frontend-owned presentation.

Shared live-helper contract
--------------------------
Pass an active ``plots`` collection and a nonempty ``name``. With ``axes=None``,
each helper creates and registers a new native Figure under that unused name.
With explicit axes, ``plots[name]`` must already exist. Each supplied axes must
still belong to that Figure and must not be claimed by another live handle.
Helpers add their artists without clearing existing content. See each helper
for its axes count, layout and labels.

Figure creation or axes attachment, artist initialization, optional configuration
and presentation run synchronously through the Plots host on its owner thread.
Successful initialization presents the Figure immediately, once per Figure even
when several helpers share it. Workers may hold typed handles, but must not
modify active axes or artists directly. Host and configuration errors propagate.
Initialization failure does not present a new Figure or roll back changes.

Returned handles expose ``update``. They validate and copy real-valued input to
float64 arrays before owner dispatch, then synchronously mutate artists. Invalid
dimensions, shapes or complex-valued data raise ValueError before mutation.
Updates refresh the Figure unless ``refresh=False``; use ``plots.refresh(name)``
after a batch. An update's optional ``title=None`` preserves the current title.
A non-None title replaces it on the updated plot, as described by each helper.
NaN signals can represent unmeasured samples.

Helper creation raises ValueError for an empty or conflicting new name, or for
wrong-count, repeated, foreign, removed or already-claimed axes. Explicit axes
with an unknown name raise KeyError. Creating or updating after ``plots.finish``
raises RuntimeError; creating a new Figure in a sealed collection also raises
RuntimeError. Removing template artists or their axes makes later handle updates
raise RuntimeError. Host, NumPy conversion and Matplotlib errors propagate.
Stop producers before finish, then release presentation separately. Native
figures remain usable after release.
"""

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
        """Set the red best-point overlay; NaN coordinates hide it.

        Raise RuntimeError if draw removed this marker or its original axes.
        """
        self._ensure_active()
        point = np.array([[x, y]], dtype=np.float64)

        def apply() -> None:
            self._ensure_active()
            if self._axes not in self._figure.axes:
                raise RuntimeError("Point marker was removed from its original axes.")
            if self._marker is None:
                self._marker = self._axes.scatter(
                    [], [], color="red", label="Best Point", zorder=3
                )
            elif (
                self._marker.axes is not self._axes
                or self._marker not in self._axes.collections
            ):
                raise RuntimeError("Point marker was removed from its original axes.")
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

    # tolerated: One factory supplies owner, two axes/segments and scan direction; no shared parameter group.
    def __init__(  # noqa: PLR0913
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
        """Set a red dashed reference on the scan-line axes; NaN hides it.

        Raise RuntimeError if draw removed this marker or its original axes.
        """
        self._ensure_active()
        position = float(x)

        def apply() -> None:
            self._ensure_active()
            if self._line_axes not in self._figure.axes:
                raise RuntimeError("Line marker was removed from its original axes.")
            if self._marker is None:
                self._marker = self._line_axes.axvline(
                    np.nan, color="red", linestyle="--"
                )
            elif (
                self._marker.axes is not self._line_axes
                or self._marker not in self._line_axes.lines
            ):
                raise RuntimeError("Line marker was removed from its original axes.")
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
    that producer's order. General figures stay unpresented until finish unless
    drawn. Live factories and the first successful draw present immediately. Finishing fixes membership and stops typed
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

    def fork(self) -> Plots:
        """Return an empty independent producer using this same presentation host.

        Membership, Figure ownership, typed handles, finish and release are not
        shared. The caller owns the new producer's lifecycle. Always return a
        plain Plots, even when this producer has a frontend-specific subtype.
        """
        return Plots(self._host)

    def ensure_active(self) -> None:
        """Raise RuntimeError after finish, before creating or updating artists."""
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
        self._live_axes.intersection_update(
            ax for current in self.values() for ax in current.axes
        )
        self._live_axes.update(resolved)
        return figure, resolved

    def _present_live(self, figure: Figure) -> None:
        if figure not in self._live:
            self._host.present(figure)
            self._live.append(figure)

    def create_live(
        self,
        name: str,
        initialize: Callable[
            [PlotHost, Figure, tuple[Axes, ...], Callable[[], None]], _T
        ],
        *,
        axes: tuple[Axes, ...] | None = None,
        count: int = 1,
    ) -> _T:
        """Initialize a live figure on its owner and return the callback's handle.

        Without axes, create a unique named figure with count columns. Otherwise
        claim count distinct, unused axes of self[name]. Conflicts raise
        ValueError; an unknown existing name raises KeyError. A finished
        collection raises RuntimeError. The callback receives the host, figure,
        claimed axes and an active-lifetime checker for its update handle.
        It may return a handle that retains artists for owner-managed updates,
        but workers must not mutate those artists directly.
        Present only after initialization succeeds, once per figure. Callback
        and host errors propagate without presentation or rollback.
        """
        self.ensure_active()

        def create() -> _T:
            self.ensure_active()
            figure, resolved = self._resolve_axes(name, axes, count)
            handle = initialize(self._host, figure, resolved, self.ensure_active)
            self._present_live(figure)
            return handle

        return self._host.call(create)

    def draw(self, name: str, callback: Callable[[Figure], None]) -> None:
        """Draw on an existing named figure, then refresh on the host owner.

        Execution is synchronous on the host owner thread and blocks the calling
        worker thread until callback, presentation and refresh return.
        The first successful draw presents the figure and makes it live; later
        draws refresh the same presentation. Untouched general figures still
        wait until finish to present. With a GUI host it
        occupies the GUI thread; heavy drawing can freeze the interface.
        The callback may freely modify axes shared with live templates, but must
        not retain artists for later worker-side mutation. Calling cla() in the
        callback removes same-axes template artists; their next update raises
        RuntimeError. Callback exceptions propagate back to the caller without
        rollback, presentation or refresh. Unknown names raise KeyError; finished plots raise
        RuntimeError. Host errors also propagate.
        """
        self.ensure_active()
        figure = self[name]

        def draw_frame() -> None:
            self.ensure_active()
            callback(figure)
            self._present_live(figure)
            self._host.refresh(figure)

        self._host.call(draw_frame)

    def save(self, name: str, path: str | Path) -> None:
        """Save an existing named figure on the host owner without refreshing.

        Execution is synchronous on the host owner thread and blocks the calling
        worker thread until saving completes. With a GUI host it occupies the
        GUI thread; expensive rendering can freeze the interface. This method
        invokes no callback and does not refresh. In draw(), callback exceptions
        propagate to its caller, and callback cla() invalidates same-axes
        templates whose next update raises RuntimeError; saving does not restore
        those artists.
        Use native Figure.savefig defaults and infer format from path. Saving is
        allowed before or after finish. Parent directories are not created and
        existing files follow native savefig overwrite behavior. Unknown names
        raise KeyError; host, rendering and I/O errors propagate.
        """
        figure = self[name]
        self._host.call(lambda: figure.savefig(path))

    def refresh(self, name: str) -> None:
        """Refresh a complete live frame after a batch of typed updates."""
        self.ensure_active()
        figure = self[name]
        if figure not in self._live:
            raise ValueError("Figure is not live")

        def refresh_frame() -> None:
            self.ensure_active()
            self._host.refresh(figure)

        self._host.call(refresh_frame)

    def record_animation(self, name: str, path: str | Path) -> MovieRecording:
        """Record a live figure with FFmpeg, without changing its presentation.

        The producer calls grab_frame after each complete update. Finish the
        recorder when acquisition ends; Plots.finish also closes open recorders.
        Writer errors propagate and do not declare the operation successful.
        """
        from matplotlib.animation import FFMpegWriter

        self.ensure_active()
        figure = self[name]
        if figure not in self._live:
            raise ValueError("Figure is not live")
        if figure in self._recordings:
            raise ValueError("Figure already has a movie recording")
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("FFmpeg is required to record animations")

        def start() -> MovieRecording:
            self.ensure_active()
            writer = FFMpegWriter(fps=30)
            writer.setup(figure, str(path), dpi=200)
            recording = MovieRecording(self._host, writer, self.ensure_active)
            self._recordings[figure] = recording
            return recording

        return self._host.call(start)

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
                # tolerated: Isolate arbitrary writer/host failures so every recording is tried, then report all.
                except Exception as error:  # noqa: BLE001
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
                # tolerated: Try every host release, aggregate failures and leave failed figures retryable.
                except Exception as error:  # noqa: BLE001
                    errors.append(error)
                else:
                    self._released.add(figure)
            if errors:
                raise ExceptionGroup("Failed to release plot presentation", errors)

        self._host.call(cleanup)


# tolerated: Preserve public labels, style, owner-side configuration and existing-axes options.
def live_line(  # noqa: PLR0913
    plots: Plots,
    name: str,
    xlabel: str,
    ylabel: str,
    *,
    title: str | None = None,
    num_lines: int = 1,
    configure_axes: Callable[[Axes], None] | None = None,
    axes: Axes | None = None,
) -> LinePlot:
    """Create or attach a live line plot and return a LinePlot update handle.

    Follow this module's shared live-helper contract for ``plots``, ``name``,
    owner execution, presentation, lifetime and errors. With ``axes=None``,
    create one axes in a new named Figure. Otherwise attach to the supplied
    axes of ``plots[name]``. Set its x/y labels to ``xlabel``/``ylabel`` and,
    when non-None, its initial axes title to ``title``.

    ``num_lines`` is the positive number of line artists; values below 1 raise
    ValueError. ``configure_axes``, when supplied, runs once on the host owner
    after artist initialization and before presentation. Use it to set native
    ticks or style. It must not retain active axes for later worker mutation.
    Its exceptions propagate without a new presentation or rollback.

    Call ``handle.update(xs, signals, title=None, refresh=True)`` with real 1D
    xs and signals shaped ``(num_lines, len(xs))``. For one line, signals may
    also be 1D. Each row replaces one line and the axes autoscale to the data.
    An update title replaces this axes title. Invalid data raises ValueError;
    the shared contract describes copying, refresh and RuntimeError cases.
    """
    plots.ensure_active()
    if num_lines < 1:
        raise ValueError("Line count must be positive")

    def create(
        host: PlotHost,
        figure: Figure,
        resolved: tuple[Axes, ...],
        ensure_active: Callable[[], None],
    ) -> LinePlot:
        (ax,) = resolved
        segment = Plot1DSegment(xlabel, ylabel, title=title, num_lines=num_lines)
        segment.init_ax(ax)
        if configure_axes is not None:
            configure_axes(ax)
        return LinePlot(host, figure, ax, segment, ensure_active)

    return plots.create_live(name, create, axes=None if axes is None else (axes,))


def live_scatter(
    plots: Plots,
    name: str,
    xlabel: str,
    ylabel: str,
    *,
    title: str | None = None,
    axes: Axes | None = None,
) -> ScatterPlot:
    """Create or attach scalar-colored points and return a ScatterPlot handle.

    Follow this module's shared live-helper contract for ``plots``, ``name``,
    owner execution, presentation, lifetime and errors. With ``axes=None``,
    create one axes in a new named Figure. Otherwise attach to the supplied
    axes of ``plots[name]``. Set its x/y labels to ``xlabel``/``ylabel`` and,
    when non-None, its initial axes title to ``title``.

    Call ``handle.update(xs, ys, colors=colors, title=None, refresh=True)``.
    xs, ys and colors must be nonempty real 1D arrays of equal length. Each
    update replaces all points and their scalar color coordinates, autoscales
    the axes, and rescales colors to the new color-data range. Colors are not
    RGB tuples or color names. An update title replaces this axes title.
    Invalid data raises ValueError; the shared contract describes copying,
    refresh and RuntimeError cases.
    """
    plots.ensure_active()

    def create(
        host: PlotHost,
        figure: Figure,
        resolved: tuple[Axes, ...],
        ensure_active: Callable[[], None],
    ) -> ScatterPlot:
        (ax,) = resolved
        segment = ScatterSegment(xlabel, ylabel, title=title)
        segment.init_ax(ax)
        return ScatterPlot(host, figure, ax, segment, ensure_active)

    return plots.create_live(name, create, axes=None if axes is None else (axes,))


# tolerated: Preserve public labels, grid choice, color limits and existing-axes options.
def live_heatmap(  # noqa: PLR0913
    plots: Plots,
    name: str,
    xlabel: str,
    ylabel: str,
    *,
    title: str | None = None,
    uniform: bool = True,
    clim: tuple[float, float] | None = None,
    axes: Axes | None = None,
) -> HeatmapPlot:
    """Create or attach one live heatmap and return a HeatmapPlot handle.

    Follow this module's shared live-helper contract for ``plots``, ``name``,
    owner execution, presentation, lifetime and errors. With ``axes=None``,
    create one axes in a new named Figure. Otherwise attach to the supplied
    axes of ``plots[name]``. Set its x/y labels to ``xlabel``/``ylabel`` and,
    when non-None, its initial axes title to ``title``.

    ``uniform=True`` renders an evenly spaced grid spanning the xs/ys endpoints.
    Use ``uniform=False`` for monotonic, possibly uneven sample coordinates in
    either scan direction. Both use nearest-sample rendering. ``clim=(min, max)``
    fixes the color limits across updates; ``clim=None`` autoscales each update.
    The pair supplies Matplotlib's lower and upper color limits.

    Call ``handle.update(xs, ys, signals, title=None, refresh=True)`` with
    nonempty real 1D xs and ys and real signals shaped ``(len(xs), len(ys))``.
    signals[i, j] is the value at x=xs[i], y=ys[j]. An update title replaces
    this axes title. Invalid data raises ValueError; the shared contract
    describes copying, refresh and RuntimeError cases. The handle also offers
    ``mark_point(x, y)`` for an owner-updated red reference point; NaN hides it.
    """
    plots.ensure_active()

    def create(
        host: PlotHost,
        figure: Figure,
        resolved: tuple[Axes, ...],
        ensure_active: Callable[[], None],
    ) -> HeatmapPlot:
        (ax,) = resolved
        vmin, vmax = clim if clim is not None else (None, None)
        heatmap = (
            Plot2DSegment(xlabel, ylabel, title, vmin=vmin, vmax=vmax)
            if uniform
            else PlotNonUniform2DSegment(xlabel, ylabel, title, vmin=vmin, vmax=vmax)
        )
        heatmap.init_ax(ax)
        return HeatmapPlot(host, figure, ax, heatmap, ensure_active)

    return plots.create_live(name, create, axes=None if axes is None else (axes,))


# tolerated: Preserve public scan direction/count, grid choice, labels and paired-axes options.
def live_heatmap_with_line(  # noqa: PLR0913
    plots: Plots,
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
    """Create or attach a heatmap with scan lines; return a HeatmapLinePlot.

    Follow this module's shared live-helper contract for ``plots``, ``name``,
    owner execution, presentation, lifetime and errors. With ``axes=None``,
    create a new named Figure with one row and two columns: heatmap left,
    scan lines right. Explicit ``axes`` must be two distinct axes of
    ``plots[name]``, ordered ``(heatmap_axes, line_axes)``; retain their layout.
    ``xlabel``/``ylabel`` label the heatmap's x/y axes. The scan-line x label
    is xlabel for ``line_axis=0`` or ylabel for ``line_axis=1``; its y label
    is empty. A non-None ``title`` sets both initial axes titles.

    ``line_axis=0`` plots signals[:, j] against xs; ``line_axis=1`` plots
    signals[i, :] against ys. Other values raise ValueError. ``num_lines``
    must be positive, otherwise raise ValueError. Show the final slice that
    is not all NaN as the current line, plus up to num_lines-1 preceding
    slices in scan order. Missing earlier slices remain NaN. The current
    line has point markers; earlier lines are translucent red.
    ``uniform=True`` renders an evenly spaced grid spanning xs/ys endpoints.
    Use ``uniform=False`` for monotonic, possibly uneven sample coordinates in
    either scan direction. Both use nearest-sample rendering. Heatmap colors
    autoscale on every update.

    Call ``handle.update(xs, ys, signals, title=None, refresh=True)`` with
    nonempty real 1D xs and ys and real signals shaped ``(len(xs), len(ys))``.
    signals[i, j] is the value at x=xs[i], y=ys[j]. Each update replaces the
    heatmap and scan lines; an update title changes only the heatmap title.
    Invalid data raises ValueError; the shared contract describes copying,
    refresh and RuntimeError cases. The handle also offers ``mark_line(x)``
    for an owner-updated red dashed vertical reference on the scan-line axes;
    NaN hides it.
    """
    plots.ensure_active()
    if num_lines < 1:
        raise ValueError("Line count must be positive")
    if line_axis not in (0, 1):
        raise ValueError("line_axis must be 0 or 1")

    def create(
        host: PlotHost,
        figure: Figure,
        resolved: tuple[Axes, ...],
        ensure_active: Callable[[], None],
    ) -> HeatmapLinePlot:
        heatmap_axes, line_axes = resolved
        heatmap = (
            Plot2DSegment(xlabel, ylabel, title)
            if uniform
            else PlotNonUniform2DSegment(xlabel, ylabel, title)
        )
        line_kwargs = [
            {"marker": "None", "alpha": 0.3, "color": "red"} for _ in range(num_lines)
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
        return HeatmapLinePlot(
            host,
            figure,
            heatmap_axes,
            line_axes,
            heatmap,
            lines,
            line_axis,
            ensure_active,
        )

    return plots.create_live(name, create, axes=axes, count=2)
