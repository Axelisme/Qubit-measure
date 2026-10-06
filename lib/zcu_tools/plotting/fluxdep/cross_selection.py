"""Explicit Qt/native rendering of captured cross-spectrum selection."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from zcu_tools.analysis.fluxdep.cross_selection import (
    CrossSelectionInputs,
    CrossSelectionState,
    CrossSelectionView,
    project_cross_selection,
)


class CrossSelectionPlot:
    """Reusable artists on a caller-owned Figure, shared by Qt and native output.

    Render backgrounds at sample centers (descending/nonuniform allowed),
    calibrated flux/GHz points, ticks, grid and normalized-radius stroke outlines.
    No numerical preprocessing, Session mutation, pyplot or canvas adoption.
    """

    def __init__(self, figure: Figure, inputs: CrossSelectionInputs) -> None:
        """Create the layout on figure without replacing its canvas.

        inputs supplies read-only cloud/backgrounds and joint bounds.
        Caller owns Figure lifetime; render on its canvas owner thread.
        """
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
            [], [], s=15, color="red", label="Kept", zorder=4
        )
        self._dropped = self._axes.scatter(
            [], [], s=12, color="#707070", alpha=0.6, label="Dropped", zorder=3
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
        self._axes.legend(loc="best", fontsize="small")
        figure.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.82)

    def show_state(
        self, view: CrossSelectionView, *, show_changes: bool = False
    ) -> None:
        """Render exact view, kept/dropped points and optional latest changes.

        Title shows kept/total, normalized distance/radius and radius in flux/GHz.
        Changes use hollow added points, removed crosses and counts; stroke is a
        thin dashed line with endpoint ellipses scaled by joint-bound spans.
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
            f"Selected {np.count_nonzero(result.selected)}/{result.selected.size}; "
            f"min_distance {state.min_distance:g}; width (radius) {state.width:g}\n"
            f"Radius = {state.width * self._x_span:g} flux / "
            f"{state.width * self._y_span:g} GHz"
        )
        if show_changes:
            info += (
                f"; added {len(view.added_points)}, removed {len(view.removed_points)}"
            )
        self._axes.set_title(info, fontsize=9)


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
