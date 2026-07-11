from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.device import device_setup_cancel_scope
from zcu_tools.experiment.v2.runner import StopSignal, schedule_stop_scope
from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    NO_RESULT,
    BgResult,
    ExclusionRequest,
    OperationRunner,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.scopes import progress_ambient

from .guard import RunPermit
from .operation_gate import OperationKind
from .scopes import figure_ambient

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .ports import RunStatePort, WritebackLifecyclePort


class RunService(QObject):
    """Encapsulates execution of an experiment adapter via BackgroundRunner
    (OffMain-thread strategy with figure/progress/cancel scopes — ADR-0019).

    Uses OperationRunner (ADR-0026 §1) for the lifecycle mechanism; domain
    policy (cancel-partial interpretation, State writes, signals) is inline here.
    """

    run_finished: Signal = Signal(str, object)
    run_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: RunStatePort,
        runner: OperationRunner,
        bus: EventBus,
        handles: OperationHandles,
        writeback: WritebackLifecyclePort,
    ) -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._bus = bus
        # handles is used by cancel_run (handles.cancel) and for live_count queries
        # by the controller. The runner also holds a reference to the same handles
        # instance (shared, single source of truth for operation lifecycle).
        self._handles = handles
        self._writeback = writeback
        # active_token is set by begin() and cleared on the terminal path so the
        # controller can cancel_run() and await the outcome (ADR-0019).
        self._active_token: int | None = None

    def start_run(
        self,
        permit: RunPermit,
        live_container: FigureContainer | None = None,
    ) -> int:
        # Static preconditions (context readiness, committed-cfg validity, soc
        # capability) are proven by the RunPermit. Dynamic resource availability
        # (tab busy, hardware exclusion) is checked here at the operation boundary.
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        logger.info("start_run: tab_id=%r", tab_id)

        # PRE-OPEN: Starting a run invalidates the previous run/analyze/writeback
        # result. Drop them before begin() so the tab honestly has "no result this
        # run" while it is in flight (and if it fails/cancels). Tear down the per-
        # item writeback editor models first (State requires it).
        self._writeback.teardown_tab_items(tab_id)
        self._state.clear_tab_results(tab_id)

        # A single StopSignal owns the Schedule-visible stop flag for this run.
        # ``cancel_requested`` is separate: Schedule failures also set the stop
        # flag to halt host loops, but they are not user cancellations.
        stop_event = threading.Event()
        stop_signal = StopSignal(stop_event)
        cancel_requested = threading.Event()
        adapter = permit.adapter
        request = permit.request
        schema = permit.schema

        def request_cancel() -> None:
            cancel_requested.set()
            stop_event.set()

        def work(factory: Any) -> Any:
            # Run is the OffMain-thread strategy with all three scopes (ADR-0026 §2):
            # figure routing+liveplot (figure_ambient, app layer), progress
            # (progress_ambient, session layer), and cancel (Schedule StopSignal
            # plus device setup cancel scope).
            with figure_ambient(live_container):
                with progress_ambient(factory):
                    with schedule_stop_scope(stop_signal):
                        with device_setup_cancel_scope(stop_event):
                            result = adapter.run(request, schema)
                            stop_signal.raise_if_error()
                            return result

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            # Interpret bg outcome: we own stop_event, so we decide cancelled vs
            # finished/failed (ADR-0019). Mirrors the old _on_bg_done/_on_bg_error
            # → _on_run_finished/_on_run_cancelled/_on_run_failed logic exactly.
            if bg.ok:
                if cancel_requested.is_set() or stop_event.is_set():
                    _on_run_cancelled(bg.result, settle)
                else:
                    _on_run_finished(bg.result, settle)
            else:
                assert bg.error is not None
                if cancel_requested.is_set() and stop_signal.error is None:
                    _on_run_cancelled(NO_RESULT, settle)
                else:
                    _on_run_failed(bg.error, settle)

        def _on_run_finished(result: Any, settle: SettleFn) -> None:
            logger.info(
                "_on_run_finished: tab_id=%r result_type=%s",
                tab_id,
                type(result).__name__,
            )
            # New run invalidates the previous analyze's writeback draft: tear down
            # before update_tab_result clears the list.
            self._writeback.teardown_tab_items(tab_id)
            self._state.update_tab_result(tab_id, result)
            self._state.set_tab_running(tab_id, False)
            self._active_token = None
            # STATE is observable before settle (ADR-0017 / stage2c invariant 1).
            settle(OperationOutcome("finished"))
            self._bus.emit(RunFinishedPayload(tab_id=tab_id, outcome="finished"))
            self.run_finished.emit(tab_id, result)

        def _on_run_cancelled(result: Any, settle: SettleFn) -> None:
            logger.info("_on_run_cancelled: tab_id=%r", tab_id)
            # A cancelled run may still carry a partial result (the worker returned
            # before the stop_event tripped a hard interrupt); keep it if present.
            if result is not NO_RESULT:
                self._writeback.teardown_tab_items(tab_id)
                self._state.update_tab_result(tab_id, result)
            self._state.set_tab_running(tab_id, False)
            self._active_token = None
            # STATE is observable before settle.
            settle(OperationOutcome("cancelled"))
            self._bus.emit(RunFinishedPayload(tab_id=tab_id, outcome="cancelled"))
            # A cancelled run emits run_finished only if it produced a usable result.
            if result is not NO_RESULT:
                self.run_finished.emit(tab_id, result)

        def _on_run_failed(error: Exception, settle: SettleFn) -> None:
            logger.warning("_on_run_failed: tab_id=%r error=%r", tab_id, error)
            self._state.set_tab_running(tab_id, False)
            self._active_token = None
            # STATE is observable before settle.
            settle(OperationOutcome("failed", str(error)))
            self._bus.emit(
                RunFinishedPayload(
                    tab_id=tab_id,
                    outcome="failed",
                    error_message=str(error),
                )
            )
            self.run_failed.emit(tab_id, error)

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.RUN,
                owner_id=tab_id,
            ),
            owner_id=tab_id,
            wants_progress=True,
            cancel_hook=request_cancel,
            work=work,
            run_in_pool=False,
            on_terminal=on_terminal,
        )

        # begin() is atomic: ensure_can_start → create → register → factory → submit.
        # Raises on conflict or submit-fail (settle unwinds resources inside runner).
        try:
            token = self._runner.begin(spec)
        except Exception:
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id=tab_id,
                    fact=TabInteractionFact.RUN_START_REJECTED,
                )
            )
            raise

        # POST-BEGIN: tab is marked running and started events are emitted only
        # after begin() succeeds (a begin-raise means no worker started — ADR-0026).
        self._active_token = token
        self._state.set_tab_running(tab_id, True)
        with self._bus.origin(self._handles.event_origin(token)):
            self._bus.emit(RunStartedPayload(tab_id=tab_id))
        return token

    @property
    def active_token(self) -> int | None:
        """The token of the currently running operation, or None."""
        return self._active_token

    def cancel_run(self) -> bool:
        """Request cancellation of the active run; best-effort.

        Returns True when a live run token existed and was signalled (the request
        was issued), False when no run was in flight (a graceful no-op). This is
        NOT a claim that the worker has stopped: the worker self-judges 'cancelled'
        and emits its terminal asynchronously (ADR-0019) — the true terminal is
        observed via gui_op_wait/poll on the run handle.
        """
        logger.info("cancel_run")
        # Async notification: set the operation's stop_event via the handle.
        token = self._active_token
        if token is None:
            return False
        self._handles.cancel(token)
        return True
