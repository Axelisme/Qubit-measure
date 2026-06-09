"""Shared interactive flux-line-pick analysis for the flux_dep adapters.

Both onetone/flux_dep and twotone/flux_dep produce a 2D map whose half-flux /
integer-flux sweet-spot lines the USER picks by dragging — there is no automatic
fit. This module is their shared ``AnalysisMode.INTERACTIVE`` implementation:

- ``FluxPickParams``: the analyze-params marker — these adapters expose no tunable
  analyze params (the magnitude-only projection is fixed per adapter, not on the form).
- ``FluxPickResult``: the deferred result (flx_half / flx_int / flx_period), built
  on the user's Done — flows through the same path as a FIT analyze result.
- ``FluxPickSession``: the ``InteractiveSession`` wrapping the toolkit-agnostic
  ``TwoLinePicker`` on the host's figure; it offloads the heavy auto-align step to
  ``host.run_background`` and repaints via ``host.redraw``.

The adapters call ``build_flux_pick_session(req, host, force_magnitude=...)`` from
``setup_interactive_analysis`` (passing their fixed projection) and return the
flx_* writeback items from ``get_writeback_items``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from matplotlib.figure import Figure

from zcu_tools.gui.app.main.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    InteractiveHost,
    InteractiveSession,
)
from zcu_tools.notebook.analysis.fluxdep.interactive.two_line_picker import (
    TwoLinePicker,
)


@dataclass
class FluxPickParams:
    """Analyze-params marker for the interactive flux-pick adapters.

    These adapters expose no tunable analyze params: the magnitude-only spectrum
    projection is fixed per adapter (one-tone True — phase uninformative; two-tone
    False — phase may carry signal) and passed straight to
    ``build_flux_pick_session`` by each adapter, never surfaced on the form.
    """


@dataclass
class FluxPickResult(AnalyzeResultBase):
    flx_half: float
    flx_int: float
    flx_period: float
    figure: Optional[Figure] = None


class FluxPickSession:
    """``InteractiveSession`` driving a ``TwoLinePicker`` on the host's figure."""

    def __init__(self, picker: TwoLinePicker, host: InteractiveHost) -> None:
        self._picker = picker
        self._host = host

    # pointer events: mutate the (passive) picker, then repaint
    def on_press(self, x: Optional[float]) -> None:
        self._picker.on_press(x)
        self._host.redraw()

    def on_move(self, x: Optional[float]) -> None:
        self._picker.on_move(x)
        self._host.redraw()

    def on_release(self, x: Optional[float], y: Optional[float]) -> None:
        self._picker.on_release(x, y)
        self._host.redraw()

    def actions(self) -> "list[tuple[str, str]]":
        return [("auto_align", "Auto Align"), ("swap", "Swap Lines")]

    def invoke_action(self, action_id: str) -> None:
        if action_id == "auto_align":
            # Heavy (a mirror-loss search): compute off the main thread, then apply
            # + repaint back on the main thread.
            self._host.run_background(
                self._picker.compute_aligned_positions,
                self._apply_aligned,
            )
        elif action_id == "swap":
            self._picker.swap()
            self._host.redraw()
        else:
            raise ValueError(f"unknown action {action_id!r}")

    def _apply_aligned(self, positions: object) -> None:
        half, integer = positions  # type: ignore[misc]
        self._picker.apply_positions(half, integer)
        self._host.redraw()

    def info_text(self) -> str:
        return self._picker.info_text()

    def finish(self) -> FluxPickResult:
        half, integer = self._picker.positions()
        return FluxPickResult(
            flx_half=half,
            flx_int=integer,
            flx_period=2 * abs(integer - half),
            figure=self._host.figure,
        )


def build_flux_pick_session(
    req: AnalyzeRequest[Any, Any], host: InteractiveHost, *, force_magnitude: bool
) -> FluxPickSession:
    """Build the flux-pick session: a ``TwoLinePicker`` on the host figure, seeded
    from any previously-calibrated flx_half / flx_int in the MetaDict."""
    result = req.run_result
    seed_half = req.md.get("flx_half", None)
    seed_int = req.md.get("flx_int", None)
    picker = TwoLinePicker(
        host.figure,
        result.signals,
        result.values,
        result.freqs,
        flux_half=seed_half,
        flux_int=seed_int,
        force_magnitude=force_magnitude,
    )
    return FluxPickSession(picker, host)
