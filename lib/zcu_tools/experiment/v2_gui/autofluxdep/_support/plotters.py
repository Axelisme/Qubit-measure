"""Shared Plotters for the autofluxdep Builders, aligned with the runner module.

A Plotter is a Node-type-defined, stateful object drawing that Node's figure on
the main thread. Its lifetime is the whole Sweep (built once at Run start, fed
the flux-aware Result as it fills, redrawn each flux point). It holds drawing
state but never owns the Qt widget, and is NEVER marshalled — the worker only
fills numpy rows + notifies (ADR-0067).

Each Plotter uses typed factories on the run-owned named ``Plots`` collection.
The UI calls updates on the main thread and refreshes the attached canvas after
all panels have changed. Presentation never owns the retained Figure.

Three shapes cover the experiments (qubit_freq keeps its own two-panel Plotter):

- ``Decay1DPlotter`` — t1 / t2ramsey / t2echo: a (flux → fitted scalar) scatter +
  the *current* flux point's signal-vs-axis trace with the fitted curve.
- ``ColormapLinePlotter`` — lenrabi / mist: a flux × axis colormap with the latest
  flux rows as 1-D traces, plus an optional red marker.
- ``Landscape2DPlotter`` — ro_optimize: the current flux point's freq × gain
  landscape (HeatmapPlot) with the best (freq, gain) marked.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from zcu_tools.experiment.v2_gui.autofluxdep._support.result import (
    Sweep1DResult,
    Sweep2DResult,
)
from zcu_tools.plotting.plots import Plots


class Decay1DPlotter:
    """t1 / t2ramsey / t2echo: flux→scalar scatter + the current point's curve.

    Two LinePlot panels (matching the runner's ``t1`` / ``t1_curve``):
    - top: flux value → fitted scalar (t1 / t2r / t2e), drawn as markers only.
    - bottom: the current flux point's signal vs the swept axis, with the fitted
      curve overlaid (so the user sees how well each point fit).
    """

    def __init__(
        self, plots: Plots, figure_name: str, title: str, value_label: str, x_label: str
    ) -> None:
        figure = plots[figure_name]
        self._fig = figure
        ax_scalar = figure.add_subplot(2, 1, 1)
        ax_curve = figure.add_subplot(2, 1, 2)
        self._curve_title = f"{title} (curve)"

        def configure_scalar(ax: Axes) -> None:
            ax.lines[0].set_linestyle("None")
            ax.lines[0].set_marker(".")

        self._scalar = plots.liveplot_1d(
            figure_name,
            "Flux device value",
            value_label,
            axes=ax_scalar,
            title=f"{title} ({value_label})",
            configure_axes=configure_scalar,
        )
        # the current point's signal (x = swept axis) + fitted curve as 2 lines
        self._curve = plots.liveplot_1d(
            figure_name,
            x_label,
            "Signal (a.u.)",
            axes=ax_curve,
            title=self._curve_title,
            num_lines=2,
        )

    def update(self, result: Sweep1DResult, idx: int) -> None:
        self._scalar.update(result.flux, result.fit_value, refresh=False)
        # the current flux row's raw signal + fitted curve, stacked as 2 lines
        n = result.n_flux
        i = idx if 0 <= idx < n else n - 1
        two = np.vstack([result.signal[i], result.fit_curve[i]])
        self._curve.update(
            result.x,
            two,
            title=title_with_snr(self._curve_title, result, i),
            refresh=False,
        )
        self._fig.canvas.draw_idle()


class ColormapLinePlotter:
    """lenrabi / mist: a flux × axis colormap + the latest flux rows as traces.

    One HeatmapLinePlot (matching the runner's ``rabi_curve`` / ``mist``): the
    2-D map is flux × swept-axis, the side panel shows the latest ``num_lines``
    flux rows. An optional red dashed line marks a tracked value (lenrabi's
    pi_length); ``marker_of`` extracts it per update (None = no marker, e.g.
    mist).
    """

    def __init__(
        self,
        plots: Plots,
        figure_name: str,
        title: str,
        y_label: str,
        num_lines: int = 3,
        marker_of: Callable[[Sweep1DResult], float] | None = None,
    ) -> None:
        figure = plots[figure_name]
        self._fig = figure
        self._title = title
        ax_2d = figure.add_subplot(1, 2, 1)
        ax_line = figure.add_subplot(1, 2, 2)
        self._marker_of = marker_of
        self._plot = plots.liveplot_2d_with_line(
            figure_name,
            "Flux device value",
            y_label,
            line_axis=1,
            num_lines=num_lines,
            title=title,
            axes=(ax_2d, ax_line),
        )

    def update(self, result: Sweep1DResult, idx: int) -> None:
        self._plot.update(
            result.flux,
            result.x,
            result.signal,
            title=title_with_snr(self._title, result, idx),
            refresh=False,
        )
        if self._marker_of is not None:
            self._plot.mark_line(float(self._marker_of(result)))
        self._fig.canvas.draw_idle()


class Landscape2DPlotter:
    """ro_optimize: the current flux point's freq × gain landscape + best marker.

    One HeatmapPlot (matching the runner's ``snr``): only the current flux row's
    freq × gain map is shown (a 3-D volume cannot be one image), with the argmax
    (best_freq, best_gain) marked by a red point.
    """

    def __init__(
        self, plots: Plots, figure_name: str, title: str = "ro_optimize"
    ) -> None:
        figure = plots[figure_name]
        self._fig = figure
        ax = figure.add_subplot(1, 1, 1)
        self._plot = plots.liveplot_2d(
            figure_name,
            "Frequency (MHz)",
            "Gain (a.u.)",
            axes=ax,
            title=title,
        )
        self._best = ax.scatter(
            [np.nan],
            [np.nan],
            color="red",
            edgecolors="white",
            linewidths=0.8,
            label="Latest best",
            zorder=3,
        )

    def update(self, result: Sweep2DResult, idx: int) -> None:
        n = result.n_flux
        i = idx if 0 <= idx < n else n - 1
        self._plot.update(result.freq, result.gain, result.signal[i], refresh=False)
        best = _latest_best_offset(result, i)
        self._best.set_offsets(best)
        self._fig.canvas.draw_idle()


def _latest_best_offset(result: Sweep2DResult, idx: int) -> NDArray[np.float64]:
    """Return the latest finite best point at or before ``idx`` as scatter offsets."""
    if result.n_flux == 0:
        return np.array([[np.nan, np.nan]], dtype=np.float64)
    hi = min(max(idx, 0), result.n_flux - 1) + 1
    finite = np.isfinite(result.best_freq[:hi]) & np.isfinite(result.best_gain[:hi])
    if not np.any(finite):
        return np.array([[np.nan, np.nan]], dtype=np.float64)
    best_idx = int(np.flatnonzero(finite)[-1])
    return np.array(
        [[float(result.best_freq[best_idx]), float(result.best_gain[best_idx])]],
        dtype=np.float64,
    )


def title_with_snr(base: str, result: Any, idx: int) -> str:
    snr = getattr(result, "snr", None)
    if snr is None:
        return base
    values = np.asarray(snr, dtype=np.float64)
    if values.size == 0:
        return base
    i = idx if 0 <= idx < values.shape[0] else values.shape[0] - 1
    value = float(values[i])
    if not np.isfinite(value) or value <= 0.0:
        return base
    return f"{base} (snr = {value:.1f})"
