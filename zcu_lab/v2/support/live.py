"""Caller-thread artist tools for lab experiments.

Helpers attach to the given axes without creating figures, setting labels or
titles, dispatching, refreshing or presenting. Their set methods validate data
and return changed artists. Call set inside @plots.on_plot_thread after display.
Authors coordinate shared artists, autoscale and color limits. The tools do not
bind a Plots lifetime; removed artists make subsequent set calls fail.
"""

from matplotlib.axes import Axes

from ._live_attached import AttachedHeatmap, AttachedLine, AttachedScatter


def live_line(axes: Axes, *, num_lines: int = 1) -> AttachedLine:
    """Attach num_lines local lines; set(xs, signals) returns changed artists.

    num_lines must be positive. AttachedLine documents accepted data shapes,
    validation errors and caller-thread mutation without refresh or dispatch.
    Labels, title, ticks and grid remain unchanged.
    """
    return AttachedLine(axes, num_lines=num_lines)


def live_scatter(axes: Axes) -> AttachedScatter:
    """Attach local scalar-colored points, preserving labels, title and grid.

    AttachedScatter.set accepts nonempty real xs/ys/colors vectors of equal
    length and returns changed artists. Invalid data raises ValueError before
    mutation; Matplotlib errors propagate. No refresh or dispatch is performed.
    """
    return AttachedScatter(axes)


def live_heatmap(
    axes: Axes,
    *,
    uniform: bool = True,
    clim: tuple[float, float] | None = None,
) -> AttachedHeatmap:
    """Attach a local nearest-sample image without setting labels or title.

    uniform=True spans endpoints evenly; False accepts monotonic uneven centers
    in either direction. clim fixes (min, max); None rescales to finite data.
    AttachedHeatmap.set validates real nonempty 1D axes and signals shaped
    (len(xs), len(ys)) before mutation and returns the image artist. No refresh
    or dispatch is performed; Matplotlib errors propagate.
    """
    return AttachedHeatmap(axes, uniform=uniform, clim=clim)
