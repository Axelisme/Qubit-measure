"""Shared Plotters for the autofluxdep Builders, aligned with the runner module.

A Plotter is a Node-type-defined, stateful object drawing that Node's figure on
the main thread. Its lifetime is the whole Sweep (built once at Run start, fed
the flux-aware Result as it fills, redrawn each flux point). It holds drawing
state but never owns the Qt widget, and is NEVER marshalled — the worker only
fills numpy rows + notifies (ADR-0017).

Each Plotter reuses ``zcu_tools.liveplot`` (LivePlot1D / LivePlot2DwithLine /
LivePlot2D) embedded into the host Figure's axes via ``existed_axes`` — the same
plot classes and the same per-subplot data the runner module's
``update_plotter`` feeds, so the GUI liveplot matches the notebook exactly. With
``existed_axes`` the liveplot owns no figure (``fig is None``); the host
refreshes by calling ``canvas.draw_idle()`` after ``update(..., refresh=False)``
(see ``zcu_tools.liveplot.segments.base``).

Three shapes cover the experiments (qubit_freq keeps its own two-panel Plotter):

- ``Decay1DPlotter`` — t1 / t2ramsey / t2echo: a (flux → fitted scalar) scatter +
  the *current* flux point's signal-vs-axis trace with the fitted curve.
- ``ColormapLinePlotter`` — lenrabi / mist: a flux × axis colormap (LivePlot2D-
  withLine) with the latest flux rows as 1-D traces, plus an optional red marker.
- ``Landscape2DPlotter`` — ro_optimize: the current flux point's freq × gain
  landscape (LivePlot2D) with the best (freq, gain) marked.
"""

from __future__ import annotations

import numpy as np
from typing_extensions import Any, Optional

from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult, Sweep2DResult


class Decay1DPlotter:
    """t1 / t2ramsey / t2echo: flux→scalar scatter + the current point's curve.

    Two LivePlot1D panels (matching the runner's ``t1`` / ``t1_curve``):
    - top: flux value → fitted scalar (t1 / t2r / t2e), drawn as markers only.
    - bottom: the current flux point's signal vs the swept axis, with the fitted
      curve overlaid (so the user sees how well each point fit).
    """

    def __init__(self, figure: Any, title: str, value_label: str, x_label: str) -> None:
        from zcu_tools.liveplot import LivePlot1D

        self._fig = figure
        ax_scalar = figure.add_subplot(2, 1, 1)
        ax_curve = figure.add_subplot(2, 1, 2)
        self._scalar = LivePlot1D(
            "Flux device value",
            value_label,
            existed_axes=[[ax_scalar]],
            segment_kwargs=dict(
                title=f"{title} ({value_label})",
                line_kwargs=[dict(linestyle="None", marker=".")],
            ),
        )
        # the current point's signal (x = swept axis) + fitted curve as 2 lines
        self._curve = LivePlot1D(
            x_label,
            "Signal (a.u.)",
            existed_axes=[[ax_curve]],
            segment_kwargs=dict(title=f"{title} (curve)", num_lines=2),
        )
        self._scalar.__enter__()
        self._curve.__enter__()

    def update(self, result: Sweep1DResult, idx: int) -> None:
        self._scalar.update(result.flux, result.fit_value, refresh=False)
        # the current flux row's raw signal + fitted curve, stacked as 2 lines
        n = result.n_flux
        i = idx if 0 <= idx < n else n - 1
        two = np.vstack([result.signal[i], result.fit_curve[i]])
        self._curve.update(result.x, two, refresh=False)
        self._fig.canvas.draw_idle()


class ColormapLinePlotter:
    """lenrabi / mist: a flux × axis colormap + the latest flux rows as traces.

    One LivePlot2DwithLine (matching the runner's ``rabi_curve`` / ``mist``): the
    2-D map is flux × swept-axis, the side panel shows the latest ``num_lines``
    flux rows. An optional red dashed line marks a tracked value (lenrabi's
    pi_length); ``marker_of`` extracts it per update (None = no marker, e.g.
    mist).
    """

    def __init__(
        self,
        figure: Any,
        title: str,
        y_label: str,
        num_lines: int = 3,
        marker_of: Optional[Any] = None,
    ) -> None:
        from zcu_tools.liveplot import LivePlot2DwithLine

        self._fig = figure
        ax_2d = figure.add_subplot(1, 2, 1)
        ax_line = figure.add_subplot(1, 2, 2)
        self._marker_of = marker_of
        self._marker = (
            ax_line.axvline(np.nan, color="red", linestyle="--")
            if marker_of is not None
            else None
        )
        self._plot = LivePlot2DwithLine(
            "Flux device value",
            y_label,
            line_axis=1,
            num_lines=num_lines,
            title=title,
            existed_axes=[[ax_2d, ax_line]],
        )
        self._plot.__enter__()

    def update(self, result: Sweep1DResult, idx: int) -> None:
        del idx
        self._plot.update(result.flux, result.x, result.signal, refresh=False)
        if self._marker is not None and self._marker_of is not None:
            self._marker.set_xdata([float(self._marker_of(result))])
        self._fig.canvas.draw_idle()


class Landscape2DPlotter:
    """ro_optimize: the current flux point's freq × gain landscape + best marker.

    One LivePlot2D (matching the runner's ``snr``): only the current flux row's
    freq × gain map is shown (a 3-D volume cannot be one image), with the argmax
    (best_freq, best_gain) marked by a red point.
    """

    def __init__(self, figure: Any, title: str = "ro_optimize") -> None:
        from zcu_tools.liveplot import LivePlot2D

        self._fig = figure
        ax = figure.add_subplot(1, 1, 1)
        self._best = ax.scatter([np.nan], [np.nan], color="red", label="Best", zorder=3)
        self._plot = LivePlot2D(
            "Frequency (MHz)",
            "Gain (a.u.)",
            existed_axes=[[ax]],
            segment_kwargs=dict(title=title),
        )
        self._plot.__enter__()

    def update(self, result: Sweep2DResult, idx: int) -> None:
        n = result.n_flux
        i = idx if 0 <= idx < n else n - 1
        self._plot.update(result.freq, result.gain, result.signal[i], refresh=False)
        bf, bg = result.best_freq[i], result.best_gain[i]
        if not (np.isnan(bf) or np.isnan(bg)):
            self._best.set_offsets([[float(bf), float(bg)]])
        self._fig.canvas.draw_idle()
