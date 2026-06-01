from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.event_bus import (
    GuiEvent,
    RunFinishedPayload,
    RunStartedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.plot_host import FigureContainer
from zcu_tools.gui.services.guard import RunPermit
from zcu_tools.gui.services.operation_gate import (
    OperationGate,
    OperationKind,
    OperationLease,
    OperationOutcome,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.pbar_host import ProgressBarModel
    from zcu_tools.gui.runner import Runner
    from zcu_tools.gui.services.progress import ProgressService
    from zcu_tools.gui.services.writeback import WritebackService
    from zcu_tools.gui.state import State


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
        # Tabs whose current run was explicitly cancelled, so the terminal
        # handler can report outcome='cancelled' instead of finished/failed.
        self._cancel_requested_tabs: set[str] = set()

        self._runner.run_finished.connect(self._on_run_finished)
        self._runner.run_failed.connect(self._on_run_failed)

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

        # Symmetric release: this lease is released exactly-once on the terminal
        # path — _on_run_finished / _on_run_failed → _release_lease (covers
        # success / failure / cancel). A pre-worker failure releases in the
        # except below.
        lease = self._gate.acquire(OperationKind.RUN, owner_id=tab_id)
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

    def get_run_progress(self) -> Tuple[Tuple[int, "ProgressBarModel"], ...]:
        # Live (handle_id, ProgressBarModel) pairs for the running tab; the caller
        # reads their methods (format/percent/...) at serialization time. Owner is
        # the running tab_id, so progress follows the tab across runs.
        running = self._state.running_tab_id
        if running is None:
            return ()
        return self._progress.bars_for_owner(running)

    def cancel_run(self) -> None:
        logger.info("cancel_run")
        # Record the running tab so the terminal handler reports 'cancelled'
        # regardless of whether the worker returns a partial result or raises.
        running = self._state.running_tab_id
        if running is not None:
            self._cancel_requested_tabs.add(running)
        self._runner.cancel()

    def _on_run_finished(self, tab_id: str, result: Any) -> None:
        logger.info(
            "_on_run_finished: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        # New run invalidates the previous analyze's writeback draft: tear down
        # its per-item editor models before update_tab_result clears the list.
        self._writeback.teardown_tab_items(tab_id)
        self._state.update_tab_result(tab_id, result)
        self._state.set_tab_running(tab_id, False)
        was_cancelled = tab_id in self._cancel_requested_tabs
        self._release_lease(
            OperationOutcome("cancelled" if was_cancelled else "finished")
        )
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._cancel_requested_tabs.discard(tab_id)
        self._bus.emit(
            GuiEvent.RUN_FINISHED,
            RunFinishedPayload(
                tab_id=tab_id,
                outcome="cancelled" if was_cancelled else "finished",
            ),
        )
        self.run_finished.emit(tab_id, result)

    def _on_run_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_run_failed: tab_id=%r error=%r", tab_id, error)
        self._state.set_tab_running(tab_id, False)
        was_cancelled = tab_id in self._cancel_requested_tabs
        self._release_lease(
            OperationOutcome(
                "cancelled" if was_cancelled else "failed",
                None if was_cancelled else str(error),
            )
        )
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._cancel_requested_tabs.discard(tab_id)
        self._bus.emit(
            GuiEvent.RUN_FINISHED,
            RunFinishedPayload(
                tab_id=tab_id,
                outcome="cancelled" if was_cancelled else "failed",
                error_message=None if was_cancelled else str(error),
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
