from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.app.main.event_bus import TabInteractionChangedPayload
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome

from .background import BackgroundService, OffMainScopes
from .guard import AnalyzePermit

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import InteractiveSession
    from zcu_tools.gui.app.main.event_bus import EventBus
    from zcu_tools.gui.app.main.state import State

    from .writeback import WritebackService


class AnalyzeService(QObject):
    analyze_finished: Signal = Signal(str, object)
    analyze_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: "State",
        bg: BackgroundService,
        bus: "EventBus",
        writeback: "WritebackService",
        handles: OperationHandles,
    ) -> None:
        super().__init__()
        self._state = state
        self._bg = bg
        self._bus = bus
        self._writeback = writeback
        # FIT analyze is the OffMain-thread strategy with only the figure-routing
        # scope (no progress, no cancel). It takes **only a Handle, no exclusion**
        # (ADR-0019): analyze never conflicts with hardware, so it no longer fakes
        # an exclusion lease just to obtain the async handle (operation_id + await).
        # The handle is settled exactly once on the terminal slot (_on_analyze_
        # finished / _failed).
        self._handles = handles
        self._active_token: Optional[int] = None

    def _release(self, outcome: OperationOutcome) -> None:
        token = self._active_token
        if token is not None:
            self._active_token = None
            self._handles.settle(token, outcome)

    def start_analyze(
        self,
        permit: AnalyzePermit,
        analyze_params_instance: object,
        figure_container: Optional[FigureContainer] = None,
    ) -> int:
        # Context + run-result preconditions are proven by the AnalyzePermit;
        # tab-busy is the dynamic check that stays at the operation boundary.
        # Returns the operation token (handle) so the caller can await it.
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        req = AnalyzeRequest(
            run_result=tab.run_result,
            analyze_params=analyze_params_instance,
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
        logger.info(
            "start_analyze: tab_id=%r analyze_params_type=%s",
            tab_id,
            type(analyze_params_instance).__name__,
        )
        # Handle only — no exclusion, no stop_event (analyze never conflicts and
        # is not cancellable in this minimal integration). Settled exactly once on
        # the terminal slot, or here if the worker fails to start.
        token = self._handles.create()
        self._active_token = token
        adapter = tab.adapter
        scopes = OffMainScopes(figure_container=figure_container)
        try:
            self._bg.submit(
                lambda: adapter.analyze(req),
                scopes,
                run_in_pool=False,
                on_done=lambda result: self._on_analyze_finished(tab_id, result),
                on_error=lambda exc: self._on_analyze_failed(tab_id, exc),
            )
        except Exception:
            self._release(OperationOutcome("failed", "analyze failed to start"))
            raise
        self._state.set_tab_analyzing(tab_id, True)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        return token

    def start_interactive(self, permit: AnalyzePermit) -> int:
        """Begin an INTERACTIVE analysis: open the async handle and mark the tab
        analyzing. There is NO worker — the View mounts the interactive canvas on
        the main thread and the user paces the work (Main-thread-user-paced
        strategy, ADR-0019); the handle is held until ``finish_interactive``
        (Done). Returns the operation token (handle)."""
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")
        token = self._handles.create()
        self._active_token = token
        self._state.set_tab_analyzing(tab_id, True)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        return token

    def finish_interactive(self, tab_id: str, session: "InteractiveSession") -> None:
        """The user finished the interactive pick (Done): build the result and run
        the SAME terminal path as a FIT analyze (writeback compute + State update +
        lease release + events), so the agent's analyze-result poll resolves."""
        self._on_analyze_finished(tab_id, session.finish())

    def _on_analyze_finished(self, tab_id: str, analyze_result: Any) -> None:
        logger.info(
            "_on_analyze_finished: tab_id=%r result_type=%s",
            tab_id,
            type(analyze_result).__name__,
        )
        # Tear down the previous analyze's writeback editor models before the new
        # draft replaces them (ADR-0008: per-item gc=False models are tied to a
        # specific analyze result). Compute the fresh persistent draft from the
        # new result (passed in, not written to State early), then commit result
        # + figure + items through the single mutator (which bumps the version).
        self._writeback.teardown_tab_items(tab_id)
        items = self._writeback.compute_items_for_tab(tab_id, analyze_result)
        self._state.update_tab_analyze(
            tab_id, analyze_result, analyze_result.figure, writeback_items=items
        )
        self._state.set_tab_analyzing(tab_id, False)
        # Figure + result are now in State (above), so the handle settles only
        # after they are observable — an awaiter that wakes on this sees a ready
        # figure. Release before emitting so a synchronous awaiter unblocks.
        self._release(OperationOutcome("finished"))
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self.analyze_finished.emit(tab_id, analyze_result)

    def _on_analyze_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_analyze_failed: tab_id=%r error=%r", tab_id, error)
        self._state.set_tab_analyzing(tab_id, False)
        self._release(OperationOutcome("failed", str(error)))
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self.analyze_failed.emit(tab_id, error)
