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
  ``host.run_background`` and repaints via ``host.redraw``. Controls are
  ``Conjugate Line`` (toggle), ``Auto Align`` and ``Swap Lines`` (buttons) in
  that order; the session owns the domain callback mapping.

The adapters call ``build_flux_pick_session(req, host, force_magnitude=...)`` from
``setup_interactive_analysis`` (passing their fixed projection) and return the
flx_* writeback items from ``get_writeback_items``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.figure import Figure

from zcu_tools.gui.app.main.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    InteractiveHost,
)
from zcu_tools.gui.app.main.adapter.types import (
    ButtonControl,
    ControlKey,
    InteractiveControl,
    ToggleControl,
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
    figure: Figure | None = None


class FluxPickSession:
    """``InteractiveSession`` driving a ``TwoLinePicker`` on the host's figure."""

    def __init__(self, picker: TwoLinePicker, host: InteractiveHost) -> None:
        self._picker = picker
        self._host = host
        self._finished = False
        self._cached_result: FluxPickResult | None = None

    # pointer events: mutate the (passive) picker, then repaint; ignored after finish
    def on_press(self, x: float | None) -> None:
        if self._finished:
            return
        self._picker.on_press(x)
        self._host.redraw()

    def on_move(self, x: float | None) -> None:
        if self._finished:
            return
        self._picker.on_move(x)
        self._host.redraw()

    def on_release(self, x: float | None, y: float | None) -> None:
        if self._finished:
            return
        self._picker.on_release(x, y)
        self._host.redraw()

    def controls(self) -> tuple[InteractiveControl, ...]:
        return (
            ToggleControl(
                key=ControlKey("conjugate"),
                label="Conjugate Line",
                initial=False,
                on_change=self._set_conjugate,
            ),
            ButtonControl(
                key=ControlKey("auto_align"),
                label="Auto Align",
                on_trigger=self._trigger_auto_align,
            ),
            ButtonControl(
                key=ControlKey("swap"),
                label="Swap Lines",
                on_trigger=self._trigger_swap,
            ),
        )

    def _set_conjugate(self, value: bool) -> None:
        if self._finished:
            return
        self._picker.set_conjugate(bool(value))

    def _trigger_auto_align(self) -> None:
        if self._finished:
            return
        # Heavy (a mirror-loss search): compute off the main thread, then apply
        # + repaint back on the main thread. Auto Align busy/single-flight
        # policy is unchanged — only post-finish late completion is gated here.
        self._host.run_background(
            self._picker.compute_aligned_positions,
            self._apply_aligned,
        )

    def _trigger_swap(self) -> None:
        if self._finished:
            return
        self._picker.swap()
        self._host.redraw()

    def _apply_aligned(self, positions: object) -> None:
        if self._finished:
            return
        half, integer = positions  # type: ignore[misc]
        self._picker.apply_positions(half, integer)
        self._host.redraw()

    def info_text(self) -> str:
        return self._picker.info_text()

    def finish(self) -> FluxPickResult:
        if self._cached_result is not None:
            return self._cached_result
        self._finished = True
        half, integer = self._picker.positions()
        result = FluxPickResult(
            flx_half=half,
            flx_int=integer,
            flx_period=2 * abs(integer - half),
            figure=self._host.figure,
        )
        self._cached_result = result
        return result


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
