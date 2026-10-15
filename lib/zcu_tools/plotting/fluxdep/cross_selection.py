"""Explicit Qt/native rendering of captured cross-spectrum selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.cross_selection import (
    CrossSelectionInputs,
    CrossSelectionState,
    CrossSelectionView,
    project_cross_selection,
)
from zcu_tools.analysis.fluxdep.stroke import BrushPoint

from .brush import BrushBand
from .palette import (
    ADDED_COLOR,
    ADDED_MARKER,
    DROPPED_COLOR,
    KEPT_COLOR,
    REMOVED_COLOR,
    REMOVED_MARKER,
)


class CrossSelectionPlot:
    """Reusable artists on a caller-owned Figure, shared by Qt and native output.

    Render backgrounds at sample centers (descending/nonuniform allowed),
    calibrated flux/GHz points, ticks, grid and the band a normalized-radius
    brush sweeps along a stroke. Point colours come from ``palette``.
    No numerical preprocessing, Session mutation, pyplot or canvas adoption.
    """

    def __init__(self, figure: Figure, inputs: CrossSelectionInputs) -> None:
        """Create the layout on figure without replacing its canvas.

        inputs supplies read-only cloud/backgrounds and joint bounds.
        Caller owns Figure lifetime; render on its canvas owner thread.
        """
        self._figure = figure
        self._axes = figure.add_subplot(111)
        self._x_span = inputs.flux_bound[1] - inputs.flux_bound[0]
        self._y_span = inputs.freq_bound[1] - inputs.freq_bound[0]
        for background in inputs.backgrounds:
            self._axes.pcolormesh(
                background.fluxs,
                background.freqs,
                np.ma.masked_invalid(background.real_signals.T),
                shading="nearest",
                cmap="gray_r",
                alpha=0.6,
                rasterized=True,
            )
        self._kept = self._axes.scatter(
            [], [], s=15, color=KEPT_COLOR, label="Kept", zorder=4
        )
        self._dropped = self._axes.scatter(
            [], [], s=12, color=DROPPED_COLOR, alpha=0.6, label="Dropped", zorder=3
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
        x_span, y_span = abs(self._x_span), abs(self._y_span)
        self._stroke = BrushBand(self._axes, x_span=x_span, y_span=y_span, zorder=6)
        self._pointer = BrushBand(self._axes, x_span=x_span, y_span=y_span, zorder=7)
        self._axes.set_xlabel("Calibrated flux (Φ/Φ₀)")
        self._axes.set_ylabel("Frequency (GHz)")
        self._axes.set_xlim(
            inputs.flux_bound[0] - 0.02 * self._x_span,
            inputs.flux_bound[1] + 0.02 * self._x_span,
        )
        self._axes.set_ylim(
            inputs.freq_bound[0] - 0.02 * self._y_span,
            inputs.freq_bound[1] + 0.02 * self._y_span,
        )
        self._axes.minorticks_on()
        self._axes.grid(True, alpha=0.25)
        # A fixed corner: "best" re-scores every background mesh cell on each
        # draw, which made opening the Filter view and every redraw stall.
        self._axes.legend(loc="upper right", fontsize="small")
        figure.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.82)

    def show_state(
        self, view: CrossSelectionView, *, show_changes: bool = False
    ) -> None:
        """Render exact view, kept/dropped points and optional latest changes.

        Title shows kept/total, normalized distance/radius and radius in flux/GHz;
        the figure layout is refitted so both title lines stay inside it.
        Changes use hollow added points, removed crosses and counts, and the
        band the stroke's brush swept along its whole path.
        Pure read defaults to no changes. No state, history or canvas mutation.
        """
        state, result = view.state, view.result
        points = np.column_stack((result.fluxs, result.freqs))
        self._kept.set_offsets(points[result.selected])
        self._dropped.set_offsets(points[~result.selected])
        self._added.set_offsets(view.added_points)
        self._removed.set_offsets(view.removed_points)
        self._added.set_visible(show_changes)
        self._removed.set_visible(show_changes)
        self._stroke.show(
            view.stroke_vertices if show_changes else (), view.stroke_width
        )
        info = (
            f"Selected {np.count_nonzero(result.selected)}/{result.selected.size}; "
            f"min_distance {state.min_distance:g}; width (radius) {state.width:g}\n"
            f"Radius = {state.width * self._x_span:g} flux / "
            f"{state.width * self._y_span:g} GHz"
        )
        if show_changes:
            info += (
                f"; added {len(view.added_points)}, removed {len(view.removed_points)}"
            )
        # wrap keeps a long line inside a narrow figure; the refit then makes
        # room above the axes for every line, which may change on each update.
        self._axes.set_title(info, fontsize=9, wrap=True)
        self._figure.tight_layout()

    def show_pointer(self, vertices: Sequence[BrushPoint], width: float) -> None:
        """Show an uncommitted drag: the band its brush sweeps along vertices.

        vertices are calibrated flux/GHz; width is the normalized radius the
        drag will commit. Empty vertices clear it. The caller redraws.
        """
        self._pointer.show(vertices, width)


def make_cross_selection_figure(
    inputs: CrossSelectionInputs,
    state: CrossSelectionState,
    *,
    show_changes: bool = False,
    previous: CrossSelectionState | None = None,
) -> Figure:
    """Return a separate Agg Figure of exact state in calibrated flux/GHz.

    previous overrides the latest change for Undo-inverse feedback.
    project_cross_selection validates numeric state, raising ValueError.
    No pyplot registration, backend switching, adoption or Session commit.
    Caller owns the Figure and can save it without any Qt runtime.
    """
    view = project_cross_selection(inputs, state, previous=previous)
    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    plot = CrossSelectionPlot(figure, inputs)
    plot.show_state(view, show_changes=show_changes)
    return figure
