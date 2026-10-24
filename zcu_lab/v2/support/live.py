"""Lab live plotting tools backed by explicit owner-managed updates.

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
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from zcu_tools.plotting import LiveUpdate
from zcu_tools.plotting.plots import PlotHost, Plots

from ._live_format import (
    HeatmapFormat,
    LineFormat,
    LineStyle,
    NonUniformHeatmapFormat,
    ScatterFormat,
)
from ._live_handles import (
    HeatmapData,
    HeatmapLinePlot,
    HeatmapPlot,
    LineData,
    LinePlot,
    ScatterData,
    ScatterPlot,
)


def live_line(  # noqa: PLR0913  # tolerated: Preserve public labels, style, owner-side configuration and existing-axes options.
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
        segment = LineFormat(xlabel, ylabel, title=title, num_lines=num_lines)
        segment.init_ax(ax)
        if configure_axes is not None:
            configure_axes(ax)

        def apply(data: LineData) -> None:
            segment.update(ax, data.xs, data.signals, data.title)

        return LinePlot(LiveUpdate(host, figure, apply, ensure_active), num_lines)

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
        segment = ScatterFormat(xlabel, ylabel, title=title)
        segment.init_ax(ax)

        def apply(data: ScatterData) -> None:
            segment.update(ax, data.xs, data.ys, colors=data.colors, title=data.title)

        return ScatterPlot(LiveUpdate(host, figure, apply, ensure_active))

    return plots.create_live(name, create, axes=None if axes is None else (axes,))


def live_heatmap(  # noqa: PLR0913  # tolerated: Preserve public labels, grid choice, color limits and existing-axes options.
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
    describes copying, refresh and RuntimeError cases.
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
            HeatmapFormat(xlabel, ylabel, title, vmin=vmin, vmax=vmax)
            if uniform
            else NonUniformHeatmapFormat(xlabel, ylabel, title, vmin=vmin, vmax=vmax)
        )
        heatmap.init_ax(ax)

        def apply(data: HeatmapData) -> None:
            heatmap.update(ax, data.xs, data.ys, data.signals, data.title)

        return HeatmapPlot(LiveUpdate(host, figure, apply, ensure_active))

    return plots.create_live(name, create, axes=None if axes is None else (axes,))


def live_heatmap_with_line(  # noqa: PLR0913  # tolerated: Preserve public scan direction/count, grid choice, labels and paired-axes options.
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
    refresh and RuntimeError cases.
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
            HeatmapFormat(xlabel, ylabel, title)
            if uniform
            else NonUniformHeatmapFormat(xlabel, ylabel, title)
        )
        line_kwargs: list[LineStyle] = [
            {"marker": "None", "alpha": 0.3, "color": "red"} for _ in range(num_lines)
        ]
        line_kwargs[-1].update(
            {"label": "current line", "marker": ".", "alpha": 1.0, "color": "C0"}
        )
        lines = LineFormat(
            xlabel if line_axis == 0 else ylabel,
            "",
            title=title,
            num_lines=num_lines,
            line_kwargs=line_kwargs,
        )
        heatmap.init_ax(heatmap_axes)
        lines.init_ax(line_axes)

        def apply(data: HeatmapData) -> None:
            heatmap.update(heatmap_axes, data.xs, data.ys, data.signals, data.title)
            line_xs = data.xs if line_axis == 0 else data.ys
            valid = ~np.all(np.isnan(data.signals), axis=line_axis)
            current = int(np.flatnonzero(valid)[-1]) if np.any(valid) else -1
            line_data = np.full((num_lines, line_xs.size), np.nan)
            for offset in range(num_lines):
                index = current - offset
                if index < 0:
                    break
                line_data[-offset - 1] = (
                    data.signals[:, index] if line_axis == 0 else data.signals[index, :]
                )
            lines.update(line_axes, line_xs, line_data)

        return HeatmapLinePlot(LiveUpdate(host, figure, apply, ensure_active))

    return plots.create_live(name, create, axes=axes, count=2)
