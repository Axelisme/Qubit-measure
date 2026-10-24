"""Native device/GHz TwoTone figures shared by Qt preview and headless output."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.stroke import BrushPoint
from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickState,
    TwoTonePickView,
    project_twotone_pick,
)

from ._background import NonUniformBackground
from .brush import BrushBand
from .palette import (
    ADDED_COLOR,
    ADDED_MARKER,
    KEPT_COLOR,
    REMOVED_COLOR,
    REMOVED_MARKER,
)


class TwoTonePickPlot:
    """Project numerical views onto a supplied Figure without committing state.

    Background and mask cells follow native sample coordinates in either scan
    direction, including nonuniform axes.
    """

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
        # Brush radii use sample endpoint spans, not the padded image limits.
        self._x_span = abs(device_last - device_first)
        self._y_span = abs(frequency_last - frequency_first)
        self._background = NonUniformBackground(
            "Device value (native)", "Frequency (GHz)"
        )
        self._mask_segment = NonUniformBackground(
            "Device value (native)", "Frequency (GHz)", vmin=0, vmax=1
        )
        # NonUniformImage does not bake scalar alpha into its Agg raster.
        # The mask only displays ones, so encode its blue and opacity together.
        mask_cmap = ListedColormap([colormaps["Blues"](1.0, alpha=0.15)], name="Blues")
        for segment, cmap in (
            (self._background, "gray_r"),
            (self._mask_segment, mask_cmap),
        ):
            segment.init_ax(self._axes, cmap=cmap)
            segment.update(
                self._axes,
                spectrum.dev_values,
                spectrum.freqs,
                np.zeros(spectrum.signals.shape),
            )
        # Freeze limits before overlays can autoscale against placeholder data.
        self._axes.set_xlim(self._axes.get_xlim())
        self._axes.set_ylim(self._axes.get_ylim())
        self._mask = self._axes.get_images()[-1]
        self._mask.set_visible(False)
        self._current = self._axes.scatter(
            [], [], s=12, color=KEPT_COLOR, label="Current", zorder=4
        )
        self._added = self._axes.scatter(
            [],
            [],
            s=45,
            marker=ADDED_MARKER,
            edgecolors=ADDED_COLOR,
            facecolors="none",
            label="Added",
            zorder=5,
        )
        self._removed = self._axes.scatter(
            [],
            [],
            s=30,
            color=REMOVED_COLOR,
            marker=REMOVED_MARKER,
            label="Removed",
            zorder=5,
        )
        # The latest committed stroke, then an uncommitted drag above it.
        x_span, y_span = self._x_span, self._y_span
        self._stroke = BrushBand(self._axes, x_span=x_span, y_span=y_span, zorder=6)
        self._pointer = BrushBand(self._axes, x_span=x_span, y_span=y_span, zorder=7)
        self._axes.set_xlabel("Device value (native)")
        self._axes.set_ylabel("Frequency (GHz)")
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
        adds separate added/removed markers and the band the stroke's brush
        swept along its whole path. show_mask overlays selection.
        show_origin=False masks the background only. The figure layout is
        refitted so both title lines stay inside it. Never modify view or its
        domain/history.
        """
        state, result = view.state, view.result
        background = (
            result.real_signals
            if show_origin
            else np.where(state.mask, result.real_signals, np.nan)
        )
        finite = result.real_signals[np.isfinite(result.real_signals)]
        if finite.size:
            self._background.vmin = float(np.min(finite))
            self._background.vmax = max(
                float(np.max(finite)), float(np.min(finite)) + 1e-12
            )
        spectrum = self._inputs.spectrum
        self._background.update(
            self._axes, spectrum.dev_values, spectrum.freqs, background
        )
        self._mask_segment.update(
            self._axes,
            spectrum.dev_values,
            spectrum.freqs,
            np.where(state.mask, 1.0, np.nan),
        )
        self._mask.set_visible(show_mask)
        self._current.set_offsets(np.column_stack((result.dev_values, result.freqs)))
        self._added.set_offsets(view.added_points)
        self._removed.set_offsets(view.removed_points)
        self._added.set_visible(show_changes)
        self._removed.set_visible(show_changes)
        self._stroke.show(
            view.stroke_vertices if show_changes else (), view.stroke_width
        )
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
        # wrap keeps a long line inside a narrow figure; the refit then makes
        # room above the axes for every line, which may change on each update.
        self._axes.set_title(info, fontsize=9, wrap=True)
        self._figure.tight_layout()

    def show_pointer(self, vertices: Sequence[BrushPoint], width: float) -> None:
        """Show an uncommitted drag: the band its brush sweeps along vertices.

        vertices are native device/GHz; width is the normalized radius the
        drag will commit. Empty vertices clear it. The caller redraws.
        """
        self._pointer.show(vertices, width)


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
