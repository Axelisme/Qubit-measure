"""Native device/GHz TwoTone figures shared by Qt preview and headless output."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickState,
    TwoTonePickView,
    project_twotone_pick,
)


class TwoTonePickPlot:
    """Project numerical views onto a supplied Figure without committing state."""

    def __init__(self, figure: Figure, inputs: TwoToneInputs) -> None:
        """Create native device/GHz spectrum artists; preserve the Figure canvas.

        inputs supplies immutable axes for extents. No pyplot registration,
        presentation, analysis state publication or Qt canvas ownership occurs.
        """
        self._figure = figure
        self._inputs = inputs
        self._axes = figure.add_subplot(111)
        spectrum = inputs.spectrum
        device_first = float(spectrum.dev_values[0])
        device_last = float(spectrum.dev_values[-1])
        frequency_first = float(spectrum.freqs[0])
        frequency_last = float(spectrum.freqs[-1])
        device_step = (device_last - device_first) / (spectrum.dev_values.size - 1)
        frequency_step = (frequency_last - frequency_first) / (spectrum.freqs.size - 1)
        # imshow extents are cell edges; signed half-steps keep sample centers aligned.
        self._extent = (
            device_first - device_step / 2,
            device_last + device_step / 2,
            frequency_first - frequency_step / 2,
            frequency_last + frequency_step / 2,
        )
        # Brush radii use sample endpoint spans, not the padded image extent.
        self._x_span = abs(device_last - device_first)
        self._y_span = abs(frequency_last - frequency_first)
        self._image = self._axes.imshow(
            np.zeros(spectrum.signals.T.shape),
            extent=self._extent,
            origin="lower",
            aspect="auto",
            cmap="gray_r",
            interpolation="nearest",
        )
        self._mask = self._axes.imshow(
            np.zeros(spectrum.signals.T.shape),
            extent=self._extent,
            origin="lower",
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=1,
            alpha=0.15,
            interpolation="nearest",
            visible=False,
        )
        self._current = self._axes.scatter(
            [], [], s=12, color="red", label="Current", zorder=4
        )
        self._added = self._axes.scatter(
            [],
            [],
            s=45,
            edgecolors="#008b6b",
            facecolors="none",
            label="Added",
            zorder=5,
        )
        self._removed = self._axes.scatter(
            [], [], s=30, color="#b04bb3", marker="x", label="Removed", zorder=5
        )
        (self._stroke,) = self._axes.plot(
            [], [], color="#007e91", linewidth=1, linestyle="--", zorder=6
        )
        self._endpoints: list[Ellipse] = []
        self._axes.set_xlabel("Device value (native)")
        self._axes.set_ylabel("Frequency (GHz)")
        self._axes.set_xlim(self._extent[:2])
        self._axes.set_ylim(self._extent[2:])
        self._axes.minorticks_on()
        self._axes.grid(True, alpha=0.25)
        self._axes.legend(loc="upper right", fontsize="small")
        figure.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.82)

    def show_state(
        self,
        view: TwoTonePickView,
        *,
        show_changes: bool = False,
        show_mask: bool = False,
        show_origin: bool = True,
    ) -> None:
        """Project exact derived result and optional latest-change overlays.

        Show native device/GHz axes, ticks/grid and current points. show_changes
        adds separate added/removed markers, stroke and normalized endpoint
        outlines. show_mask overlays selection. show_origin=False masks the
        background only. Never modify view or its domain/history.
        """
        state, result = view.state, view.result
        background = (
            result.real_signals
            if show_origin
            else np.where(state.mask, result.real_signals, np.nan)
        )
        self._image.set_data(background.T)
        finite = result.real_signals[np.isfinite(result.real_signals)]
        if finite.size:
            self._image.set_clim(
                float(np.min(finite)),
                max(float(np.max(finite)), float(np.min(finite)) + 1e-12),
            )
        self._mask.set_data(
            np.ma.masked_where(~state.mask.T, np.ones(state.mask.T.shape))
        )
        self._mask.set_visible(show_mask)
        self._current.set_offsets(np.column_stack((result.dev_values, result.freqs)))
        self._added.set_offsets(view.added_points)
        self._removed.set_offsets(view.removed_points)
        self._added.set_visible(show_changes)
        self._removed.set_visible(show_changes)
        for endpoint in self._endpoints:
            endpoint.remove()
        self._endpoints.clear()
        vertices = view.stroke_vertices if show_changes else ()
        self._stroke.set_data([p.x for p in vertices], [p.y for p in vertices])
        if vertices:
            for point in (vertices[0], vertices[-1]):
                endpoint = Ellipse(
                    (point.x, point.y),
                    width=2 * view.stroke_width * self._x_span,
                    height=2 * view.stroke_width * self._y_span,
                    fill=False,
                    edgecolor="#007e91",
                    linewidth=0.8,
                    zorder=6,
                )
                self._axes.add_patch(endpoint)
                self._endpoints.append(endpoint)
        info = (
            f"Radius {state.width:g} = {state.width * self._x_span:g} device / "
            f"{state.width * self._y_span:g} GHz; mask {np.count_nonzero(state.mask)}/{state.mask.size}\n"
            f"Points {result.dev_values.size}"
        )
        if show_changes:
            info += (
                f" (+{len(view.added_points)}, -{len(view.removed_points)}); "
                f"mask +{view.mask_added}, -{view.mask_removed}"
            )
        self._axes.set_title(info, fontsize=9)


def make_twotone_pick_figure(
    inputs: TwoToneInputs,
    state: TwoTonePickState,
    *,
    show_changes: bool = False,
    show_mask: bool = False,
    show_origin: bool = True,
    previous: TwoTonePickState | None = None,
) -> Figure:
    """Return a separate Agg Figure of exact state and optional latest changes.

    Compute via the numerical projection owner; previous overrides stored
    before-image for changes such as Undo. Propagate ValueError for invalid state. Do not register with pyplot or mutate state/history/canvas.
    """
    view = project_twotone_pick(inputs, state, previous=previous)
    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    plot = TwoTonePickPlot(figure, inputs)
    plot.show_state(
        view, show_changes=show_changes, show_mask=show_mask, show_origin=show_origin
    )
    return figure
