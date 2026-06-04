from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.event_bus import (
    GuiEvent,
    RunFinishedPayload,
    RunStartedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.app.main.plot_host import FigureContainer
from zcu_tools.gui.app.main.runner import NO_RESULT

from .guard import RunPermit
from .operation_gate import (
    OperationGate,
    OperationKind,
    OperationLease,
    OperationOutcome,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.event_bus import EventBus
    from zcu_tools.gui.app.main.runner import Runner
    from zcu_tools.gui.app.main.state import State

    from .progress import ProgressService
    from .writeback import WritebackService


class RunService(QObject):
    """Encapsulates execution of an experiment adapter via a Runner."""

    run_finished: Signal = Signal(str, object)
    run_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: "State",
        runner: "Runner",
        bus: "EventBus",
        gate: OperationGate,
        writeback: "WritebackService",
        progress: "ProgressService",
    ) -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._bus = bus
        self._gate = gate
        self._writeback = writeback
        self._progress = progress
        self._active_lease: Optional[OperationLease] = None

        self._runner.run_finished.connect(self._on_run_finished)
        self._runner.run_failed.connect(self._on_run_failed)
        self._runner.run_cancelled.connect(self._on_run_cancelled)

    def start_run(
        self,
        permit: RunPermit,
        live_container: Optional[FigureContainer] = None,
    ) -> int:
        # Static preconditions (context readiness, committed-cfg validity, soc
        # capability) are proven by the RunPermit. Dynamic resource availability
        # (tab busy, hardware exclusion) is checked here at the operation
        # boundary — see docs/adr/0001. Returns the operation token (handle).
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        logger.info("start_run: tab_id=%r", tab_id)

        # Starting a run invalidates the previous run/analyze/writeback result:
        # drop them now so the tab honestly has "no result this run" while it is
        # in flight (and if it fails/cancels). analyze/save then fail-fast with
        # the true no_run_result reason instead of reporting a misleading
        # "no params"/"no result" from behind a stale previous result. Tear down
        # the per-item writeback editor models first (State requires it).
        self._writeback.teardown_tab_items(tab_id)
        self._state.clear_tab_results(tab_id)

        # Symmetric release: this lease is released exactly-once on the terminal
        # path — _on_run_finished / _on_run_failed / _on_run_cancelled →
        # _release_lease. A pre-worker failure releases in the except below.
        # The stop_event is created here (single owner) and passed to both the
        # gate (which sets it on cancel) and the worker (which polls/self-judges).
        stop_event = threading.Event()
        lease = self._gate.acquire(
            OperationKind.RUN, owner_id=tab_id, stop_event=stop_event
        )
        self._active_lease = lease
        # Progress factory is bound to this operation (owner = tab_id) only after
        # the lease is minted; the worker gets it via the pbar ContextVar.
        pbar_factory = self._progress.make_factory(lease.token, owner_id=tab_id)
        try:
            self._runner.start_run(
                tab_id,
                permit.adapter,
                permit.request,
                permit.schema,
                stop_event,
                pbar_factory=pbar_factory,
                figure_container=live_container,
            )
        except Exception:
            self._active_lease = None
            # The run never started; settle the handle as failed and drop progress.
            self._progress.discard_operation(lease.token)
            self._gate.release(lease, OperationOutcome("failed", "run failed to start"))
            raise
        self._state.set_tab_running(tab_id, True)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(GuiEvent.RUN_STARTED, RunStartedPayload(tab_id=tab_id))
        return lease.token

    def cancel_run(self) -> None:
        logger.info("cancel_run")
        # Async notification: set the operation's stop_event via the gate and
        # return. The worker self-judges 'cancelled' (stop_event set) and emits
        # run_cancelled — no external "who I cancelled" bookkeeping needed.
        lease = self._active_lease
        if lease is not None:
            self._gate.cancel(lease.token)

    def _on_run_finished(self, tab_id: str, result: Any) -> None:
        logger.info(
            "_on_run_finished: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        # New run invalidates the previous analyze's writeback draft: tear down
        # its per-item editor models before update_tab_result clears the list.
        self._writeback.teardown_tab_items(tab_id)
        self._state.update_tab_result(tab_id, result)
        self._state.set_tab_running(tab_id, False)
        self._release_lease(OperationOutcome("finished"))
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
            GuiEvent.RUN_FINISHED,
            RunFinishedPayload(tab_id=tab_id, outcome="finished"),
        )
        self.run_finished.emit(tab_id, result)

    def _on_run_cancelled(self, tab_id: str, result: Any) -> None:
        logger.info("_on_run_cancelled: tab_id=%r", tab_id)
        # A cancelled run may still carry a partial result (the worker returned
        # before the stop_event tripped a hard interrupt); keep it if present.
        if result is not NO_RESULT:
            self._writeback.teardown_tab_items(tab_id)
            self._state.update_tab_result(tab_id, result)
        self._state.set_tab_running(tab_id, False)
        self._release_lease(OperationOutcome("cancelled"))
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
            GuiEvent.RUN_FINISHED,
            RunFinishedPayload(tab_id=tab_id, outcome="cancelled"),
        )
        # A cancelled run is reported to View listeners via run_finished only if
        # it produced a usable result; otherwise it leaves no result to consume.
        if result is not NO_RESULT:
            self.run_finished.emit(tab_id, result)

    def _on_run_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_run_failed: tab_id=%r error=%r", tab_id, error)
        self._state.set_tab_running(tab_id, False)
        self._release_lease(OperationOutcome("failed", str(error)))
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
            GuiEvent.RUN_FINISHED,
            RunFinishedPayload(
                tab_id=tab_id,
                outcome="failed",
                error_message=str(error),
            ),
        )
        self.run_failed.emit(tab_id, error)

    def _release_lease(self, outcome: OperationOutcome) -> None:
        lease = self._active_lease
        if lease is None:
            raise RuntimeError("Run completed without an active operation lease")
        self._active_lease = None
        # Destroy the operation's progress container (leave=True bars never emit
        # CLOSE, so the terminal path must clear them).
        self._progress.discard_operation(lease.token)
        self._gate.release(lease, outcome)
