from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.app.main.event_bus import GuiEvent, TabInteractionChangedPayload
from zcu_tools.gui.app.main.runner import AnalyzeRunner
from zcu_tools.gui.plotting import FigureContainer

from .guard import AnalyzePermit
from .operation_gate import (
    OperationGate,
    OperationKind,
    OperationLease,
    OperationOutcome,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.event_bus import EventBus
    from zcu_tools.gui.app.main.state import State

    from .writeback import WritebackService


class AnalyzeService(QObject):
    analyze_finished: Signal = Signal(str, object)
    analyze_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: "State",
        runner: AnalyzeRunner,
        bus: "EventBus",
        writeback: "WritebackService",
        gate: OperationGate,
    ) -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._bus = bus
        self._writeback = writeback
        # Analyze runs on a worker thread (AnalyzeWorker), so it takes an
        # operation lease purely for the async handle (operation_id + await) —
        # ANALYZE never conflicts with hardware ops. The lease is released
        # exactly once on the terminal slot (_on_analyze_finished / _failed).
        self._gate = gate
        self._active_lease: Optional[OperationLease] = None

        self._runner.analyze_finished.connect(self._on_analyze_finished)
        self._runner.analyze_failed.connect(self._on_analyze_failed)

    def _release(self, outcome: OperationOutcome) -> None:
        lease = self._active_lease
        if lease is not None:
            self._active_lease = None
            self._gate.release(lease, outcome)

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
        # Lease for the async handle only (ANALYZE never conflicts); no stop_event
        # (analyze is not cancellable in this minimal integration). Released
        # exactly once on the terminal slot, or here if the worker fails to start.
        lease = self._gate.acquire(OperationKind.ANALYZE, owner_id=tab_id)
        self._active_lease = lease
        try:
            self._runner.start_analyze(
                tab_id,
                tab.adapter,
                req,
                figure_container=figure_container,
            )
        except Exception:
            self._release(OperationOutcome("failed", "analyze failed to start"))
            raise
        self._state.set_tab_analyzing(tab_id, True)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        return lease.token

    def _on_analyze_finished(self, tab_id: str, analyze_result: Any) -> None:
        logger.info(
            "_on_analyze_finished: tab_id=%r result_type=%s",
            tab_id,
            type(analyze_result).__name__,
        )
        # Tear down the previous analyze's writeback editor models before the new
        # draft replaces them (ADR-0010: per-item gc=False models are tied to a
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
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self.analyze_finished.emit(tab_id, analyze_result)

    def _on_analyze_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_analyze_failed: tab_id=%r error=%r", tab_id, error)
        self._state.set_tab_analyzing(tab_id, False)
        self._release(OperationOutcome("failed", str(error)))
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self.analyze_failed.emit(tab_id, error)
