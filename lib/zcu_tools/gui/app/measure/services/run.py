from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from zcu_tools.datafile import RunMetadata
from zcu_tools.device.base import BaseDevice
from zcu_tools.experiment.acquisition import (
    capture_device_snapshot,
    capture_software_provenance,
    new_run_id,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.measure.events.tab import (
    TabClosedPayload,
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import CommandOrigin
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
from zcu_tools.resources.entry import AcquiredPayload, LedgerEvent, PointStorageBinding

from .guard import RunInputs
from .operation_gate import OperationKind
from .plot_lifecycle import discard_unpublished_plots, release_retired_plots

logger = logging.getLogger(__name__)
_GUI_ORIGIN = CommandOrigin("gui")

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.ports import ExclusionGate
    from zcu_tools.gui.session.types import ParameterContext
    from zcu_tools.plotting.plots import Plots

    from ..state import RetiredPaneResources, State
    from .ports import WritebackLifecyclePort


@dataclass(slots=True)
class _RunOperation:
    """One run owns its inputs and worker-produced evidence.

    Identity and start time are fixed before worker submission. The worker
    captures provenance before acquisition and completes metadata afterward.
    Terminal publishes it only with usable data; failed empty Runs persist nothing.
    """

    tab_id: str
    plots: Plots
    stop_event: threading.Event
    stop_signal: StopSignal
    cancel_requested: threading.Event
    origin: CommandOrigin
    adapter_name: str
    storage: PointStorageBinding | None
    parameters: ParameterContext | None
    metadata: RunMetadata | None = None


class RunService:
    """Encapsulates execution of an experiment adapter via BackgroundRunner
    (OffMain-thread strategy with explicit RunContext and ambient progress — ADR-0066).

    Uses OperationRunner (ADR-0066) for the lifecycle mechanism; domain
    policy (cancel-partial interpretation, State writes, facts) stays here.
    """

    def __init__(  # noqa: PLR0913 - explicit operation owners and borrowed driver source
        self,
        state: State,
        runner: OperationRunner,
        bus: EventBus,
        handles: OperationHandles,
        writeback: WritebackLifecyclePort,
        *,
        gate: ExclusionGate,
        devices: Callable[[], Mapping[str, BaseDevice[Any]]],
    ) -> None:
        self._state = state
        self._devices = devices
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
            if self._active_token is None:
                raise RuntimeError("Run result has no active operation")
            retired = self._state.update_tab_result(
                op.tab_id,
                result,
                source_operation_id=self._active_token,
                metadata=op.metadata,
                parameters=op.parameters,
            )
        except Exception as error:  # noqa: BLE001 - settle host/State failures on the owner loop
            self._run_failed(op, error, settle)
            return False
        self._view_plots[op.tab_id] = op.plots
        self._teardown_retired(retired)
        try:
            self._append_acquired(op, result)
        except Exception as error:
            logger.exception("Acquired ledger append failed: tab_id=%r", op.tab_id)
            # Publication succeeded; keep its result and view resources for Save.
            self._clear_running(op.tab_id)
            settle(OperationOutcome("failed", str(error)))
            self._bus.emit(
                RunFinishedPayload(
                    tab_id=op.tab_id,
                    outcome="failed",
                    error_message=str(error),
                )
            )
            return False
        return True

    def _append_acquired(self, op: _RunOperation, result: object) -> None:
        if result is NO_RESULT:
            return
        if op.storage is None:
            logger.warning(
                "Acquired omitted: tab_id=%r has no saved-point storage "
                "(unbound, missing storage or no native evidence)",
                op.tab_id,
            )
            return
        metadata = op.metadata
        if metadata is None:
            logger.warning(
                "Acquired omitted: tab_id=%r has no native acquisition evidence",
                op.tab_id,
            )
            return
        if not isinstance(result, RunRecord) or result.cfg is None:
            raise ValueError("Acquired requires a native RunRecord with cfg")
        if metadata.finished_at is None:
            raise ValueError("Acquired requires an acquisition finish timestamp")
        op.storage.ledger.append(
            LedgerEvent(
                id=str(uuid4()),
                kind="acquired",
                at=metadata.finished_at,
                entry_id=str(metadata.snapshot.entry_id),
                origin=op.origin.origin,
                payload=AcquiredPayload(
                    run_id=metadata.run_id,
                    tab=op.adapter_name,
                    tab_id=op.tab_id,
                    experiment=metadata.experiment,
                    cfg_summary=result.cfg.model_dump(mode="json"),
                    point=metadata.snapshot.point,
                    roles=dict(metadata.snapshot.roles),
                ),
            )
        )

    def _clear_running(self, tab_id: str) -> None:
        self._state.set_tab_running(tab_id, running=False)
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
        self._state.set_tab_running(tab_id, running=True)
        try:
            retired = self._state.clear_tab_results(tab_id)
            self._teardown_retired(retired)
        except Exception:
            self._state.set_tab_running(tab_id, running=False)
            raise

    def start_run(
        self,
        inputs: RunInputs,
        *,
        plots: Plots,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> int:
        """Start the admitted Run and return its asynchronous operation token.

        inputs must come from GuardService.capture_run_inputs and fix the exact
        cfg, source, alias names and saved-point storage. Guard does not read
        hardware. After the operation acquires its hardware lease, the worker
        reads get_info once for every connected device before adapter.run.
        Any snapshot read failure prevents adapter execution, without hardware
        writes or cache fallback; no devices produces {}. The fixed snapshot
        stays with this Run's metadata for Save, not later sweep settings.
        plots is this Run's owned host. Call on State's owner thread after
        admission.
        origin attributes acquired to the initiating gui/mcp frontend. A usable
        native result publishes to State before terminal appends acquired,
        without Save or creating a run folder. Missing storage/native evidence
        leaves Run usable but records a diagnostic, not a fabricated event.
        Ledger errors settle failed with ERROR diagnostics, retaining the
        published result and metadata for Save, without retry or rollback.
        Busy, exclusion and submission failures raise before a token is returned.
        """
        # Admission checks precede this call; inputs are data, not a proof.
        # Busy and hardware exclusion remain operation-boundary checks.
        tab_id = inputs.tab_id
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
        adapter = inputs.adapter
        request = inputs.request
        cfg = inputs.accepted_cfg.cfg
        context = RunContext(
            soc=request.soc,
            soccfg=request.soccfg,
            plots=plots,
            devices=self._devices(),
            cancel_signal=stop_signal,
        )

        def request_cancel() -> None:
            cancel_requested.set()
            stop_event.set()

        operation = _RunOperation(
            tab_id=tab_id,
            plots=plots,
            stop_event=stop_event,
            stop_signal=stop_signal,
            cancel_requested=cancel_requested,
            origin=origin,
            adapter_name=inputs.adapter_name,
            storage=inputs.storage,
            parameters=inputs.parameters,
        )

        # Fix admission identity before submission, even when the worker is delayed.
        started_at = datetime.now(UTC)
        snapshot, experiment_tag = inputs.snapshot, inputs.experiment_tag
        run_id = (
            new_run_id(started_at)
            if snapshot is not None and experiment_tag is not None
            else None
        )

        def work(factory: Any) -> object:
            # OperationRunner has acquired the hardware lease before submission.
            # A failed read rejects acquisition, including adapters without metadata.
            device_snapshot = capture_device_snapshot(context.devices)
            # Provenance uses only software/host and the cached SoC config.
            if (
                snapshot is not None
                and experiment_tag is not None
                and run_id is not None
            ):
                operation.metadata = RunMetadata(
                    run_id=run_id,
                    experiment=experiment_tag,
                    started_at=started_at.isoformat(),
                    finished_at=None,
                    completion="partial",
                    labber_path=None,
                    snapshot=snapshot,
                    provenance=capture_software_provenance(context=context),
                    device_snapshot=device_snapshot,
                )
            # Progress alone is ambient; execution dependencies are run-owned.
            with progress_ambient(factory):
                result = adapter.run(request, cfg, context=context)
                stop_signal.raise_if_error()
            # Completion records acquisition progress, not later operation cancellation.
            completion = "stopped" if stop_event.is_set() else "complete"
            finished_at = datetime.now(UTC)
            if operation.metadata is not None:
                if not isinstance(result, RunRecord):
                    raise TypeError("Native acquisition must return a RunRecord")
                operation.metadata = replace(
                    operation.metadata,
                    finished_at=finished_at.isoformat(),
                    completion=completion,
                )
            return result

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.RUN,
                owner_id=tab_id,
                note=f"run {inputs.adapter_name} (tab {tab_id})",
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
            self._state.set_tab_running(tab_id, running=False)
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
