"""Native device/GHz TwoTone figures shared by Qt preview and headless output."""

from __future__ import annotations

from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickState,
    TwoTonePickView,
)


class TwoTonePickPlot:
    """Project numerical views onto a supplied Figure without committing state."""

    def __init__(self, figure: Figure, inputs: TwoToneInputs) -> None:
        """Create native device/GHz spectrum artists; preserve the Figure canvas.

        inputs supplies immutable axes for extents. No pyplot registration,
        presentation, analysis state publication or Qt canvas ownership occurs.
        """
        raise NotImplementedError

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
        raise NotImplementedError


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
    raise NotImplementedError
