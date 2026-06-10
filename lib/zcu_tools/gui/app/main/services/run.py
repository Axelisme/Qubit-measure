from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.ports import BackgroundExecutor, ExclusionGate, ProgressHub

from .background import NO_RESULT, OffMainScopes
from .guard import RunPermit
from .operation_gate import OperationKind

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .ports import WritebackLifecyclePort


class RunService(QObject):
    """Encapsulates execution of an experiment adapter via BackgroundService
    (OffMain-thread strategy with figure/progress/cancel scopes — ADR-0019)."""

    run_finished: Signal = Signal(str, object)
    run_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: State,
        bg: BackgroundExecutor,
        bus: EventBus,
        gate: ExclusionGate,
        handles: OperationHandles,
        writeback: WritebackLifecyclePort,
        progress: ProgressHub,
    ) -> None:
        super().__init__()
        self._state = state
        self._bg = bg
        self._bus = bus
        # Run composes both leaves (ADR-0019): a Handle (operation_id + await +
        # cancel) AND an Exclusion lease (RUN conflicts with soc/device ops),
        # under one token.
        self._gate = gate
        self._handles = handles
        self._writeback = writeback
        self._progress = progress
        self._active_token: int | None = None

    def start_run(
        self,
        permit: RunPermit,
        live_container: FigureContainer | None = None,
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

        # Compose the two leaves (ADR-0019): fail-fast on hardware conflict
        # *before* opening a handle (so a conflict leaves nothing behind), then
        # mint the handle (holding the stop_event) and register the exclusion
        # under that token. Released exactly-once on the terminal path —
        # _on_run_finished / _failed / _cancelled → _release_lease; a pre-worker
        # failure unwinds in the except below. The stop_event is created here
        # (single owner): the handle sets it on cancel, the worker self-judges.
        self._gate.ensure_can_start(OperationKind.RUN)
        stop_event = threading.Event()
        token = self._handles.create(stop_event=stop_event)
        self._gate.register(token, OperationKind.RUN, owner_id=tab_id)
        self._active_token = token
        # Progress factory is bound to this operation (owner = tab_id) only after
        # the token is minted; the worker gets it via the pbar ContextVar.
        pbar_factory = self._progress.make_factory(token, owner_id=tab_id)
        # Run is the OffMain-thread strategy with all three scopes (ADR-0019):
        # figure routing+liveplot, progress, and cancel (ActiveTask). bg only
        # reports done/failed; cancellation is *interpreted* here (we own the
        # stop_event) — see _on_bg_done / _on_bg_error.
        adapter = permit.adapter
        request = permit.request
        schema = permit.schema
        scopes = OffMainScopes(
            figure_container=live_container,
            pbar_factory=pbar_factory,
            stop_event=stop_event,
        )
        try:
            self._bg.submit(
                lambda: adapter.run(request, schema),
                scopes,
                run_in_pool=False,
                on_done=lambda result: self._on_bg_done(tab_id, stop_event, result),
                on_error=lambda exc: self._on_bg_error(tab_id, stop_event, exc),
            )
        except Exception:
            self._active_token = None
            # The run never started: drop progress, settle the handle as failed,
            # free the exclusion.
            self._progress.discard_operation(token)
            self._handles.settle(
                token, OperationOutcome("failed", "run failed to start")
            )
            self._gate.release(token)
            raise
        self._state.set_tab_running(tab_id, True)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(RunStartedPayload(tab_id=tab_id))
        return token

    def cancel_run(self) -> None:
        logger.info("cancel_run")
        # Async notification: set the operation's stop_event via the handle and
        # return. The worker self-judges 'cancelled' (stop_event set) and emits
        # run_cancelled — no external "who I cancelled" bookkeeping needed.
        token = self._active_token
        if token is not None:
            self._handles.cancel(token)

    # ------------------------------------------------------------------
    # bg outcome -> cancel interpretation (we own the stop_event, ADR-0019)
    # ------------------------------------------------------------------

    def _on_bg_done(
        self, tab_id: str, stop_event: threading.Event, result: Any
    ) -> None:
        # A cancelled run may still return a partial result before the stop_event
        # tripped a hard interrupt: stop set + normal return -> cancelled(partial).
        if stop_event.is_set():
            self._on_run_cancelled(tab_id, result)
        else:
            self._on_run_finished(tab_id, result)

    def _on_bg_error(
        self, tab_id: str, stop_event: threading.Event, error: Exception
    ) -> None:
        # stop set + raise -> the run interrupted itself; cancelled with no result.
        if stop_event.is_set():
            self._on_run_cancelled(tab_id, NO_RESULT)
        else:
            self._on_run_failed(tab_id, error)

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
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
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
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
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
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
            RunFinishedPayload(
                tab_id=tab_id,
                outcome="failed",
                error_message=str(error),
            ),
        )
        self.run_failed.emit(tab_id, error)

    def _release_lease(self, outcome: OperationOutcome) -> None:
        token = self._active_token
        if token is None:
            raise RuntimeError("Run completed without an active operation token")
        self._active_token = None
        # Destroy the operation's progress container (leave=True bars never emit
        # CLOSE, so the terminal path must clear them), then settle the handle
        # (wakes the awaiter — State is already updated by the caller) and free
        # the exclusion.
        self._progress.discard_operation(token)
        self._handles.settle(token, outcome)
        self._gate.release(token)
