from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zcu_tools.device import device_setup_cancel_scope
from zcu_tools.experiment.v2.runtime import StopSignal, schedule_stop_scope
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.measure.events.tab import (
    TabClosedPayload,
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
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
from .plot_lifecycle import discard_unpublished_plots, release_retired_plots

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.ports import ExclusionGate
    from zcu_tools.plotting.plots import Plots

    from ..state import RetiredPaneResources
    from .ports import RunStatePort, WritebackLifecyclePort


@dataclass(frozen=True, slots=True)
class _RunOperation:
    """Inputs captured for one terminal callback, separate from the next run."""

    tab_id: str
    plots: Plots
    stop_event: threading.Event
    stop_signal: StopSignal
    cancel_requested: threading.Event


class RunService:
    """Encapsulates execution of an experiment adapter via BackgroundRunner
    (OffMain-thread strategy with explicit plots and progress/cancel scopes — ADR-0066).

    Uses OperationRunner (ADR-0066) for the lifecycle mechanism; domain
    policy (cancel-partial interpretation, State writes, facts) stays here.
    """

    def __init__(
        self,
        state: RunStatePort,
        runner: OperationRunner,
        bus: EventBus,
        handles: OperationHandles,
        writeback: WritebackLifecyclePort,
        *,
        gate: ExclusionGate,
    ) -> None:
        self._state = state
        self._runner = runner
        self._gate = gate
        self._bus = bus
        # handles is used by cancel_run (handles.cancel) and for live_count queries
        # by the controller. The runner also holds a reference to the same handles
        # instance (shared, single source of truth for operation lifecycle).
        self._handles = handles
        self._writeback = writeback
        # active_token is set by begin() and cleared on the terminal path so the
        # controller can cancel_run() and await the outcome (ADR-0066).
        self._active_token: int | None = None
        # Run plots are view-only, never part of the canonical pane or Save All.
        self._view_plots: dict[str, Plots] = {}
        self._bus.subscribe(TabClosedPayload, self._on_tab_closed)

    def _on_tab_closed(self, payload: TabClosedPayload) -> None:
        self.release_view_plots(payload.tab_id)

    def release_view_plots(self, tab_id: str) -> None:
        previous = self._view_plots.pop(tab_id, None)
        if previous is not None:
            try:
                previous.release()
            except Exception:
                logger.exception("Retired run view release failed: tab_id=%r", tab_id)

    def _teardown_retired(self, retired: RetiredPaneResources) -> None:
        for draft in retired.writeback_drafts:
            try:
                self._writeback.teardown_draft(draft)
            except Exception:
                logger.exception("retired run draft teardown failed")
        release_retired_plots(retired)

    def _discard_plots(self, op: _RunOperation, context: str) -> None:
        try:
            discard_unpublished_plots(op.plots)
        except Exception:
            logger.exception(
                "%s run plot cleanup failed: tab_id=%r", context, op.tab_id
            )

    def _on_run_terminal(
        self, op: _RunOperation, bg: BgResult, settle: SettleFn
    ) -> None:
        # Schedule failures set the stop flag too, but are not cancellations.
        if bg.ok:
            if op.cancel_requested.is_set() or op.stop_event.is_set():
                self._run_cancelled(op, bg.result, settle)
            else:
                self._run_finished(op, bg.result, settle)
            return
        if op.cancel_requested.is_set() and op.stop_signal.error is None:
            self._run_cancelled(op, NO_RESULT, settle)
        else:
            error = bg.error or RuntimeError("Run worker failed without an error")
            self._run_failed(op, error, settle)

    def _publish_run_result(
        self, op: _RunOperation, result: Any, settle: SettleFn
    ) -> bool:
        try:
            op.plots.finish()
            retired = self._state.update_tab_result(op.tab_id, result)
        except Exception as error:  # noqa: BLE001 - settle host/State failures on the owner loop
            self._run_failed(op, error, settle)
            return False
        self._view_plots[op.tab_id] = op.plots
        self._teardown_retired(retired)
        return True

    def _clear_running(self, tab_id: str) -> None:
        self._state.set_tab_running(tab_id, False)
        self._active_token = None

    def _run_finished(self, op: _RunOperation, result: Any, settle: SettleFn) -> None:
        logger.info(
            "_on_run_finished: tab_id=%r result_type=%s",
            op.tab_id,
            type(result).__name__,
        )
        if not self._publish_run_result(op, result, settle):
            return
        self._clear_running(op.tab_id)
        # State is observable before settle (ADR-0066).
        settle(OperationOutcome("finished"))
        self._bus.emit(RunFinishedPayload(tab_id=op.tab_id, outcome="finished"))

    def _run_cancelled(self, op: _RunOperation, result: Any, settle: SettleFn) -> None:
        logger.info("_on_run_cancelled: tab_id=%r", op.tab_id)
        # A cancelled run may still carry a partial result.
        if result is NO_RESULT:
            self._discard_plots(op, "Cancelled")
        elif not self._publish_run_result(op, result, settle):
            return
        self._clear_running(op.tab_id)
        settle(OperationOutcome("cancelled"))
        self._bus.emit(RunFinishedPayload(tab_id=op.tab_id, outcome="cancelled"))

    def _run_failed(
        self, op: _RunOperation, error: Exception, settle: SettleFn
    ) -> None:
        logger.warning("_on_run_failed: tab_id=%r error=%r", op.tab_id, error)
        self._discard_plots(op, "Failed")
        self._clear_running(op.tab_id)
        settle(OperationOutcome("failed", str(error)))
        self._bus.emit(
            RunFinishedPayload(
                tab_id=op.tab_id,
                outcome="failed",
                error_message=str(error),
            )
        )

    def _prepare_tab_for_run(self, tab_id: str) -> None:
        # Reject conflicts before reserving State or clearing results. Runner
        # checks the same gate again when opening the operation.
        self._gate.ensure_can_start(OperationKind.RUN)
        # Reserve State's existing busy flag before cleanup or synchronous gate
        # notifications can reenter tab editing and closing.
        self._state.set_tab_running(tab_id, True)
        try:
            retired = self._state.clear_tab_results(tab_id)
            self._teardown_retired(retired)
        except Exception:
            self._state.set_tab_running(tab_id, False)
            raise

    def start_run(
        self,
        permit: RunPermit,
        *,
        plots: Plots,
    ) -> int:
        # Static preconditions (context readiness, committed-cfg validity, soc
        # capability) are proven by the RunPermit. Dynamic resource availability
        # (tab busy, hardware exclusion) is checked here at the operation boundary.
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

        logger.info("start_run: tab_id=%r", tab_id)
        self.release_view_plots(tab_id)

        # PRE-OPEN: Starting a run invalidates the previous run/analyze/writeback
        # result. State performs one owner-thread swap and returns every detached
        # draft; cleanup happens only after the new empty panes are committed.
        self._prepare_tab_for_run(tab_id)

        # A single StopSignal owns the Schedule-visible stop flag for this run.
        # ``cancel_requested`` is separate: Schedule failures also set the stop
        # flag to halt host loops, but they are not user cancellations.
        stop_event = threading.Event()
        stop_signal = StopSignal(stop_event)
        cancel_requested = threading.Event()
        adapter = permit.adapter
        request = permit.request
        raw_cfg = permit.accepted_cfg.values

        def request_cancel() -> None:
            cancel_requested.set()
            stop_event.set()

        def work(factory: Any) -> Any:
            # The caller supplies operation plots; progress and cancel remain
            # scoped to this worker (ADR-0066).
            with (
                progress_ambient(factory),
                schedule_stop_scope(stop_signal),
                device_setup_cancel_scope(stop_event),
            ):
                result = adapter.run(request, raw_cfg, plots=plots)
                stop_signal.raise_if_error()
                return result

        operation = _RunOperation(
            tab_id, plots, stop_event, stop_signal, cancel_requested
        )
        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.RUN,
                owner_id=tab_id,
                note=f"run {permit.adapter_name} (tab {tab_id})",
            ),
            owner_id=tab_id,
            wants_progress=True,
            cancel_hook=request_cancel,
            work=work,
            run_in_pool=False,
            on_terminal=lambda bg, settle: self._on_run_terminal(operation, bg, settle),
        )

        # begin() is atomic: ensure_can_start → create → register → factory → submit.
        # Raises on conflict or submit-fail (settle unwinds resources inside runner).
        try:
            token = self._runner.begin(spec)
        except Exception:
            self._discard_plots(operation, "Rejected")
            self._state.set_tab_running(tab_id, False)
            self._bus.emit(
                TabInteractionChangedPayload(
                    tab_id=tab_id,
                    fact=TabInteractionFact.RUN_START_REJECTED,
                )
            )
            raise

        # Started events are emitted only after begin() succeeds (ADR-0066).
        self._active_token = token
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
        and emits its terminal asynchronously (ADR-0066) — the true terminal is
        observed via the GUI's operation.await RPC or the MCP wait(op) tool.
        """
        logger.info("cancel_run")
        # Async notification: set the operation's stop_event via the handle.
        token = self._active_token
        if token is None:
            return False
        self._handles.cancel(token)
        return True
