"""Shared Plotters for the autofluxdep Builders (see CONTEXT.md).

A Plotter is a Node-type-defined, stateful object drawing that Node's figure on
the main thread. Its lifetime is the whole Sweep (built once at Run start, fed
the flux-aware Result as it fills, redrawn each flux point). It holds drawing
state (line / colormap objects) but never owns the Qt widget, and is NEVER
marshalled — the worker only fills numpy rows + notifies (ADR-0018).

Two shapes cover every 1D/2D experiment, so they are shared rather than rewritten
per Builder:

- ``Sweep1DPlotter`` — an *accumulating* (n_flux, n_x) colormap with the
  per-row fitted scalar overlaid as a tracking line. t1 / lenrabi / t2ramsey /
  t2echo / mist all use it (mist passes no fit_value, so only the colormap
  shows). Accumulate: every settled flux row plus the current one is drawn.
- ``Sweep2DPlotter`` — ro_optimize's *overwrite* plotter: only the current flux
  row's freq×gain landscape is shown, the argmax peak marked. Overwrite, because
  a 3D (flux×freq×gain) volume cannot be a single 2D image.

(qubit_freq keeps its own Plotter — its detune axis is centred on a per-row
``predict_freq`` that shifts, so its overlay is a detune offset, not a shared
scalar.)
"""

from __future__ import annotations

import numpy as np
from typing_extensions import Any

from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult, Sweep2DResult


class Sweep1DPlotter:
    """Accumulating flux×x colormap + a fitted-scalar tracking line.

    Built once at Run start with a bare matplotlib ``Figure``. ``update(result,
    idx)`` (main thread, after each row notification) redraws the whole
    accumulated colormap and overlays ``fit_value`` (the per-row fitted scalar)
    as a line tracking down the flux axis. ``title`` / ``value_label`` are the
    Builder's domain labels.
    """

    def __init__(self, figure: Any, title: str, value_label: str) -> None:
        self._fig = figure
        self._ax = figure.add_subplot(111)
        self._im = None
        self._fit_line = None
        self._value_label = value_label
        self._ax.set_ylabel("flux index")
        self._ax.set_title(title)

    def update(self, result: Sweep1DResult, idx: int) -> None:
        del idx  # the whole accumulated map is redrawn; idx is just the trigger
        x = result.x
        extent = (float(x[0]), float(x[-1]), -0.5, result.n_flux - 0.5)
        if self._im is None:
            self._im = self._ax.imshow(
                result.signal,
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )
            self._ax.set_xlabel(result.x_label)
        else:
            self._im.set_data(result.signal)
            self._im.autoscale()

        # overlay the fitted scalar (e.g. t1 / pi_length) as a tracking line,
        # only if this Builder fits one (mist leaves fit_value all-nan).
        if not np.all(np.isnan(result.fit_value)):
            rows = np.arange(result.n_flux, dtype=np.float64)
            if self._fit_line is None:
                (self._fit_line,) = self._ax.plot(
                    result.fit_value,
                    rows,
                    "r.-",
                    linewidth=1.0,
                    markersize=3,
                    label=self._value_label,
                )
                self._ax.legend(loc="upper right", fontsize="x-small")
            else:
                self._fit_line.set_data(result.fit_value, rows)
        self._fig.canvas.draw_idle()


class Sweep2DPlotter:
    """ro_optimize's overwrite plotter: the current flux row's freq×gain landscape.

    Shows only flux row ``idx`` (a 3D volume can't be one image), with the
    argmax peak marked. Redrawn fresh each flux point.
    """

    def __init__(self, figure: Any, title: str = "ro_optimize") -> None:
        self._fig = figure
        self._ax = figure.add_subplot(111)
        self._im = None
        self._peak = None
        self._title = title
        self._ax.set_xlabel("gain")
        self._ax.set_ylabel("freq (MHz)")

    def update(self, result: Sweep2DResult, idx: int) -> None:
        freq, gain = result.freq, result.gain
        extent = (float(gain[0]), float(gain[-1]), float(freq[0]), float(freq[-1]))
        landscape = result.signal[idx]
        if self._im is None:
            self._im = self._ax.imshow(
                landscape,
                aspect="auto",
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )
        else:
            self._im.set_data(landscape)
            self._im.autoscale()

        bf, bg = result.best_freq[idx], result.best_gain[idx]
        if not (np.isnan(bf) or np.isnan(bg)):
            if self._peak is None:
                (self._peak,) = self._ax.plot([bg], [bf], "rx", markersize=10)
            else:
                self._peak.set_data([bg], [bf])
        self._ax.set_title(f"{self._title} — flux {idx}")
        self._fig.canvas.draw_idle()
