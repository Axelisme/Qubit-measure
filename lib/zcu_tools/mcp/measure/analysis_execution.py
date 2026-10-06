"""Session-owned completion of one analysis operation, without replay or recovery."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from threading import Condition, Event, Lock, Thread
from typing import Any, Literal, TypedDict

from zcu_tools.mcp.core.images import validated_png
from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.operation_wait import await_operation
from zcu_tools.mcp.measure.session import GuiConnection, GuiRpcError, MeasureMcpSession

AnalysisStage = Literal["primary", "post"]
ExecutionStatus = Literal["running", "interactive", "finished", "failed", "cancelled"]
ExecutionPhase = Literal[
    "operation",
    "result_read",
    "image_save",
    "figure_read",
    "writeback_read",
    "terminal",
]
SaveStatus = Literal[
    "not_started", "not_available", "saving", "saved", "incomplete", "unknown"
]


class InvalidAnalysisValue(TypedDict):
    path: str
    reason: str


@dataclass(frozen=True)
class AnalysisResult:
    summary: Any
    params: dict[str, Any]
    operation_state: dict[str, Any]
    invalid: list[InvalidAnalysisValue]


@dataclass(frozen=True)
class SavedImage:
    figure_name: str
    image_path: str


@dataclass(frozen=True)
class ExecutionError:
    phase: ExecutionPhase
    reason: str
    message: str
    code: str | None = None


@dataclass(frozen=True)
class AnalysisStart:
    status: Literal["not_started", "unknown", "running"] = "not_started"
    reason: str | None = None


class AnalysisWriteback(TypedDict):
    """Detached GUI writeback preview for one completed analysis operation.

    has_draft: Whether the owner published a draft; false is a confirmed absence.
    items: All native candidates, including selected=false; values and targets
        stay in the GUI owner's wire representation.
    destination_context: Native destination identity and readiness at capture
        time, not an observation or authorization for a later write.
    """

    has_draft: bool
    items: list[dict[str, object]]
    destination_context: dict[str, object]


@dataclass(frozen=True)
class ExecutionSnapshot:
    """Detached progress of one session-local Primary or Post completion.

    execution: Session-local execution ID; not a persistent run ID.
    tab: GUI tab locator on the fixed connection.
    stage: primary or post, never an inferred recipe identity.
    op: Opaque session operation handle, or None before receipt.
    start: Start admission/receipt status and any rejection reason.
    status: running, interactive, finished, failed or cancelled.
    phase: Current continuation step, or terminal after completion/failure.
    cancel_requested: Latched stop intent, not a GUI outcome.
    operation_outcome: Native terminal await reply, or None while unresolved.
    params: Accepted start or result parameters, or None before capture.
    invalidated: Owner's invalidated sections after success, or None if unknown.
    result: Captured native analysis result, or None before result_read.
    interaction: Latest interaction handoff, or None if none was delivered.
    save_status: not_started, not_available, saving, saved, incomplete or unknown.
    saved_images: Confirmed persistent image paths; failures retain this prefix.
    remaining_images: Unsaved owner image names, or None before result_read.
    unconfirmed_image: Admitted save with unknown outcome, or None.
    figure: Session preview path, or None if unavailable/not delivered.
    writeback: Completed operation's preview, or None before writeback_read.
        Interactive handoff does not capture; done joins this completion.
        A failed read retains result/save/preview facts and reports error.
    error: Continuation failure with attempted phase/reason/code, or None.
    """

    execution: str
    tab: str
    stage: AnalysisStage
    op: int | None
    start: AnalysisStart = field(default_factory=AnalysisStart)
    status: ExecutionStatus = "running"
    phase: ExecutionPhase = "operation"
    cancel_requested: bool = False
    operation_outcome: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    invalidated: list[str] | None = None
    result: AnalysisResult | None = None
    interaction: dict[str, Any] | None = None
    save_status: SaveStatus = "not_started"
    saved_images: list[SavedImage] = field(default_factory=list)
    remaining_images: list[str] | None = None
    unconfirmed_image: str | None = None
    figure: str | None = None
    writeback: AnalysisWriteback | None = None
    error: ExecutionError | None = None


def _invalid_analysis_values(value: object) -> list[InvalidAnalysisValue]:
    if not isinstance(value, list):
        raise GuiRpcError("invalid analysis field reasons", reason="incompatible_wire")
    invalid: list[InvalidAnalysisValue] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("reason"), str)
        ):
            raise GuiRpcError(
                "invalid analysis field reasons", reason="incompatible_wire"
            )
        invalid.append({"path": item["path"], "reason": item["reason"]})
    return invalid


def _analysis_result(
    reply: dict[str, Any], stage: AnalysisStage
) -> tuple[AnalysisResult, list[str]]:
    state = reply.get("operation_state")
    params = reply.get("params")
    if (
        "summary" not in reply
        or not isinstance(params, dict)
        or not isinstance(state, dict)
    ):
        raise GuiRpcError("invalid analysis result reply", reason="incompatible_wire")
    pane = state.get("analysis_state" if stage == "primary" else "post_analysis_state")
    names = pane.get("figure_names") if isinstance(pane, dict) else None
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise GuiRpcError("invalid analysis figure names", reason="incompatible_wire")
    return AnalysisResult(
        summary=deepcopy(reply["summary"]),
        params=deepcopy(params),
        operation_state=deepcopy(state),
        invalid=_invalid_analysis_values(reply.get("invalid")),
    ), list(names)


class _RpcFailure(RuntimeError):
    """Keep the attempted phase distinct from the last admitted phase."""

    def __init__(self, phase: ExecutionPhase, cause: Exception) -> None:
        super().__init__(str(cause))
        self.phase: ExecutionPhase = phase
        self.cause = cause


@dataclass(frozen=True)
class CancelError:
    reason: str
    message: str
    code: str | None = None


@dataclass(frozen=True)
class GuiCancel:
    status: Literal["requested", "not_cancellable", "not_needed", "failed"]
    error: CancelError | None = None


class _ContinuationCancelled(Exception):
    """A latched intent stopped a not-yet-admitted continuation step."""


class AnalysisExecution:
    """One fixed GUI binding and detached observations of its completion."""

    def __init__(
        self,
        snapshot: ExecutionSnapshot,
        connection: GuiConnection,
        session: MeasureMcpSession,
        closed: Event,
        invalidated: list[str] | None,
    ) -> None:
        self._snapshot = snapshot
        self._connection = connection
        self._session = session
        self._session_closed = closed
        self._closed = closed
        self._invalidated = invalidated
        self._condition = Condition()
        self._images: tuple[PngImage, ...] = ()
        self._thread: Thread | None = None
        self._completion_started = False
        self._gui_cancel: GuiCancel | None = None
        self._cancel_in_flight = False

    def bind_continuation(
        self, closed: Event, *, condition: Condition | None = None
    ) -> None:
        """Bind a recipe-local close event before native start admission.

        closed is owned by the recipe driver, which sets it and calls wake before
        joining its worker. condition optionally shares the driver's local wake
        condition for progress and completion; None keeps this owner's condition.
        The session close event remains effective too. Raise ValueError after
        admission, receipt or completion start; no GUI request or background worker
        is created. Standalone callers need not bind one.
        """
        with self._condition:
            if (
                self._completion_started
                or self._snapshot.op is not None
                or self._snapshot.start.status != "not_started"
            ):
                raise ValueError("Cannot rebind an admitted analysis lifetime")
            self._closed = closed
            if condition is not None:
                self._condition = condition

    def snapshot(self) -> ExecutionSnapshot:
        with self._condition:
            return deepcopy(self._snapshot)

    def wait(self, timeout: float) -> ToolReply:
        """Wait up to timeout seconds for completion or an interactive handoff.

        Zero returns current progress. Data is the detached ExecutionSnapshot;
        images are captured previews. Failed completion is an error ToolReply,
        not a query exception. This local wait neither reads GUI nor cancels.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._snapshot.status
                    in ("interactive", "finished", "failed", "cancelled")
                ),
                timeout=timeout,
            )
            snapshot = deepcopy(self._snapshot)
            return ToolReply(
                asdict(snapshot),
                self._images,
                is_error=snapshot.status == "failed"
                or (
                    snapshot.status == "interactive"
                    and bool(
                        snapshot.interaction
                        and snapshot.interaction.get("delivery_error")
                    )
                ),
            )

    def cancel(self) -> ToolReply:
        """Latch continuation cancellation, then request stop on the original binding."""
        with self._condition:
            if self._snapshot.phase == "terminal":
                return self._cancel_reply(GuiCancel("not_needed"))
            self._snapshot = replace(
                self._snapshot, cancel_requested=True, status="running"
            )
            self._condition.notify_all()
            op = self._snapshot.op
            if op is None:
                # Intent is retained. A late receipt triggers the original-bound stop.
                return self._cancel_reply(GuiCancel("not_needed"))
            if self._cancel_in_flight:
                self._condition.wait_for(lambda: self._gui_cancel is not None)
                assert self._gui_cancel is not None
                return self._cancel_reply(self._gui_cancel)
            self._cancel_in_flight = True
        # Never hold the execution lock while waiting for RPC serialization.
        try:
            reply = self._connection.read_internal(
                "operation.cancel", {}, operation_handle=op
            )
            status = reply.get("status")
            if status == "cancelling":
                result = GuiCancel("requested")
            elif status in ("finished", "cancelled"):
                result = GuiCancel("not_needed")
            else:
                raise GuiRpcError("invalid cancel reply", reason="incompatible_wire")
        except Exception as exc:  # noqa: BLE001 - cancellation retains partial intent
            if isinstance(exc, GuiRpcError) and exc.reason == "not_cancellable":
                result = GuiCancel("not_cancellable")
            else:
                result = GuiCancel(
                    "failed",
                    CancelError(
                        (exc.reason or "cancel_failed")
                        if isinstance(exc, GuiRpcError)
                        else "cancel_failed",
                        str(exc),
                        exc.code if isinstance(exc, GuiRpcError) else None,
                    ),
                )
        with self._condition:
            self._gui_cancel = result
            self._condition.notify_all()
            return self._cancel_reply(result)

    def _cancel_reply(self, result: GuiCancel) -> ToolReply:
        return ToolReply(
            {**asdict(deepcopy(self._snapshot)), "gui_cancel": asdict(result)},
            self._images,
            is_error=result.status == "failed",
        )

    def observe_interaction(self, reply: ToolReply, *, done: bool = False) -> None:
        """Keep the latest handoff without replacing an observed completion."""
        with self._condition:
            if self._snapshot.phase != "operation":
                return
            self._images = reply.images
            self._snapshot = replace(
                self._snapshot,
                interaction=deepcopy(reply.data),
                figure=reply.data.get("figure"),
                status="running" if done else self._snapshot.status,
            )
            self._condition.notify_all()

    def admit_start(self) -> None:
        """Mark ambiguity at dispatch, not when the execution is registered."""
        with self._condition:
            if self._closed.is_set() or self._session_closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            if self._snapshot.cancel_requested:
                raise _ContinuationCancelled
            self._snapshot = replace(self._snapshot, start=AnalysisStart("unknown"))
            self._condition.notify_all()

    def observe_start(self, started: dict[str, Any]) -> None:
        """Attach a delivered receipt to the already queryable execution."""
        with self._condition:
            self._invalidated = deepcopy(started["invalidated_on_success"])
            self._snapshot = replace(
                self._snapshot,
                op=started["handle"],
                start=AnalysisStart("running"),
                params=deepcopy(started["params"]),
                status="interactive" if started.get("interactive") else "running",
            )
            self._condition.notify_all()

    def fail_start(self, exc: Exception, *, suppressed: bool = False) -> None:
        """Retain a failed start; suppressed=True marks an unsubmitted cancellation.

        Ordinary failures retain unknown dispatch unless GUI explicitly rejects
        admission. suppressed is only valid before admission with no op receipt;
        otherwise raise ValueError. It settles a between-yield cancelled start
        without calling the native cancel hook or inventing an operation outcome.
        """
        if suppressed:
            with self._condition:
                if (
                    self._snapshot.op is not None
                    or self._snapshot.start.status != "not_started"
                ):
                    raise ValueError("Cannot suppress an admitted analysis start")
                self._snapshot = replace(self._snapshot, cancel_requested=True)
        if suppressed or isinstance(exc, _ContinuationCancelled):
            self._finish()
            return
        with self._condition:
            if isinstance(exc, GuiRpcError) and exc.request_rejected:
                self._snapshot = replace(
                    self._snapshot,
                    start=AnalysisStart("not_started", exc.reason or exc.code),
                )
        self._fail(exc)

    def start(self) -> None:
        """Start the registered completion on its own background worker.

        The registry calls this under its lock, so session close cannot miss an
        admitted worker. A known operation receipt is required. Raise ValueError
        if completion already started or no receipt exists; worker start failure
        is retained as a failed snapshot rather than losing the admitted operation.
        """
        self._claim_completion()
        if self._closed.is_set() or self._session_closed.is_set():
            self._fail(GuiRpcError("MCP session is closed", reason="session_closed"))
            return
        thread = Thread(target=self._run, name=self._snapshot.execution)
        try:
            thread.start()
        except RuntimeError as exc:
            self._fail(GuiRpcError(str(exc), reason="worker_start_failed"))
        else:
            self._thread = thread

    def complete_in_current_worker(self) -> ExecutionSnapshot:
        """Observe and capture this analysis synchronously on the calling worker.

        Use only for a registry execution admitted with start_worker=False and a
        known operation receipt. This performs the same bounded operation.await,
        result/image/writeback capture and failure isolation as background start.
        Return a detached terminal snapshot, including failures and cancellation.
        The caller owns this worker's join before session PNG cleanup. Raise
        ValueError if completion already started or no operation receipt exists.
        Recipe drivers use this to avoid a second waiting worker; standalone
        analyses keep their ordinary background start.
        """
        self._claim_completion()
        self._run()
        return self.snapshot()

    def _claim_completion(self) -> None:
        with self._condition:
            if self._completion_started:
                raise ValueError("analysis completion already started")
            if self._snapshot.op is None:
                raise ValueError("analysis completion needs an operation receipt")
            self._completion_started = True

    def wake(self) -> None:
        """Wake local native-wait pacing after cancellation or lifetime close."""
        with self._condition:
            self._condition.notify_all()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _publish(self, **changes: Any) -> None:
        with self._condition:
            self._snapshot = replace(self._snapshot, **changes)
            self._condition.notify_all()

    def _admit(self, phase: ExecutionPhase, image: str | None = None) -> None:
        with self._condition:
            if self._closed.is_set() or self._session_closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            if phase != "operation" and self._snapshot.cancel_requested:
                raise _ContinuationCancelled
            self._snapshot = replace(
                self._snapshot,
                phase=phase,
                unconfirmed_image=image,
                save_status="saving"
                if image is not None
                else self._snapshot.save_status,
            )

    def _rpc(
        self,
        phase: ExecutionPhase,
        method: str,
        params: dict[str, Any],
        *,
        image: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return self._connection.send_gui_rpc(
                method,
                params,
                timeout_seconds=timeout,
                operation_handle=self._snapshot.op,
                before_send=lambda: self._admit(phase, image),
            )
        except _ContinuationCancelled:
            raise
        except Exception as exc:
            raise _RpcFailure(phase, exc) from exc

    def _fail(self, exc: Exception, phase: ExecutionPhase | None = None) -> None:
        with self._condition:
            snapshot = self._snapshot
            phase = snapshot.phase if phase is None else phase
            reason = exc.reason if isinstance(exc, GuiRpcError) else None
            code = exc.code if isinstance(exc, GuiRpcError) else None
            save_status = snapshot.save_status
            unconfirmed = snapshot.unconfirmed_image
            if phase == "image_save" and save_status != "saved":
                save_status = "incomplete"
            if unconfirmed is not None:
                # Timeouts and failed reply encoding do not settle the admitted save.
                known_rejection = (
                    code not in (None, "timeout")
                    and reason != "response_encoding_failed"
                )
                save_status = "incomplete" if known_rejection else "unknown"
                if known_rejection:
                    unconfirmed = None
            self._snapshot = replace(
                snapshot,
                status="failed",
                phase="terminal",
                save_status=save_status,
                unconfirmed_image=unconfirmed,
                error=ExecutionError(
                    phase, reason or "execution_failed", str(exc), code
                ),
            )
            self._condition.notify_all()

    def _run(self) -> None:
        # This is the worker isolation boundary: retain partial facts for every failure.
        try:
            if self.snapshot().cancel_requested:
                self.cancel()
            if not self._await_operation():
                return
            self._complete_analysis()
        except _ContinuationCancelled:
            self._finish()
        except _RpcFailure as failure:
            self._fail(failure.cause, failure.phase)
        except Exception as exc:  # noqa: BLE001 - worker boundary must publish every failure
            self._fail(exc)

    def _finish(self, **changes: Any) -> None:
        with self._condition:
            snapshot = replace(self._snapshot, **changes)
            save_status = snapshot.save_status
            if save_status == "saving" and snapshot.unconfirmed_image is None:
                save_status = "incomplete" if snapshot.remaining_images else "saved"
            self._snapshot = replace(
                snapshot,
                status="cancelled" if snapshot.cancel_requested else "finished",
                phase="terminal",
                save_status=save_status,
            )
            self._condition.notify_all()

    def _await_operation(self) -> bool:
        op = self.snapshot().op
        if op is None:
            raise ValueError("Analysis completion requires an operation receipt")
        try:
            completion = await_operation(
                self._connection,
                op,
                closed=self._closed,
                condition=self._condition,
                before_send=lambda: self._admit("operation"),
            )
        except Exception as exc:  # Translate native wait failures at this owner.
            raise _RpcFailure("operation", exc) from exc
        reply = dict(completion.native)
        self._publish(operation_outcome=reply)
        if completion.status == "failed":
            raise GuiRpcError(
                str(reply.get("error", "analysis failed")), reason="analysis_failed"
            )
        if completion.status == "cancelled":
            self._publish(status="cancelled", phase="terminal")
            return False
        self._publish(status="running", invalidated=deepcopy(self._invalidated))
        return True

    def _complete_analysis(self) -> None:
        snapshot = self.snapshot()
        pane = "analysis" if snapshot.stage == "primary" else "post_analysis"
        method = (
            "tab.get_analyze_result"
            if snapshot.stage == "primary"
            else "tab.get_post_analyze_result"
        )
        reply = self._rpc("result_read", method, {"tab_id": snapshot.tab})
        result, names = _analysis_result(reply, snapshot.stage)
        self._publish(
            result=result, params=deepcopy(result.params), remaining_images=names
        )
        saved: list[SavedImage] = []
        for index, name in enumerate(names):
            reply = self._rpc(
                "image_save",
                "tab.save_image",
                {
                    "tab_id": snapshot.tab,
                    "subtab_id": pane,
                    "figure_name": name,
                    "image_path": None,
                },
                image=name,
            )
            path = reply.get("image_path")
            if not isinstance(path, str) or not path:
                raise GuiRpcError(
                    "invalid saved image path", reason="incompatible_wire"
                )
            saved.append(SavedImage(name, path))
            self._publish(
                saved_images=list(saved),
                remaining_images=names[index + 1 :],
                unconfirmed_image=None,
            )
        self._publish(save_status="saved" if names else "not_available")
        if names:
            reply = self._rpc(
                "figure_read",
                "tab.get_figure",
                {"tab_id": snapshot.tab, "subtab_id": pane},
            )
            encoded = reply.get("png_b64")
            if not isinstance(encoded, str):
                raise GuiRpcError(
                    "invalid analysis preview reply", reason="incompatible_wire"
                )
            image = validated_png(base64.b64decode(encoded, validate=True))
            path = self._session.write_png(image.data)
            with self._condition:
                self._images = (image,)
                self._publish(figure=str(path))
        reply = self._rpc(
            "writeback_read",
            "tab.writeback_preview",
            {"tab_id": snapshot.tab, "subtab_id": pane},
        )
        self._publish(
            writeback=AnalysisWriteback(
                has_draft=reply["has_draft"],
                items=deepcopy(reply["items"]),
                destination_context=deepcopy(reply["destination_context"]),
            )
        )
        self._finish()


class AnalysisExecutions:
    """Session lifetime owner; each opaque operation gets one completion owner."""

    def __init__(self, session: MeasureMcpSession, closed: Event) -> None:
        self._session = session
        self._closed = closed
        self._lock = Lock()
        self._next_id = 1
        self._by_op: dict[int, AnalysisExecution] = {}
        self._by_id: dict[str, AnalysisExecution] = {}

    def start(
        self,
        connection: GuiConnection,
        tab: str,
        stage: AnalysisStage,
        started: dict[str, Any] | None = None,
        *,
        interaction: dict[str, Any] | None = None,
        start_worker: bool = True,
    ) -> AnalysisExecution:
        """Register one analysis completion on the fixed connection/tab/stage.

        started is the native start receipt, or None before admission. interaction
        is an already captured handoff. Retain receipts even when close wins.
        start_worker=True starts background observation when a receipt is known.
        False only registers it; its recipe worker must call the returned owner's
        complete_in_current_worker and join before session PNG cleanup. Existing
        operation IDs return their existing owner without starting another worker.
        For recipe-local close, bind_continuation before native admission; session
        close remains effective regardless of that additional lifetime.
        """
        with self._lock:
            op = started["handle"] if started is not None else None
            if op is not None and op in self._by_op:
                return self._by_op[op]
            execution = AnalysisExecution(
                ExecutionSnapshot(
                    execution=f"analysis-{self._next_id}",
                    tab=tab,
                    stage=stage,
                    op=op,
                    start=AnalysisStart("running" if op is not None else "not_started"),
                    params=deepcopy(started["params"]) if started is not None else None,
                    status="interactive"
                    if started is not None and started.get("interactive")
                    else "running",
                    interaction=deepcopy(interaction),
                ),
                connection,
                self._session,
                self._closed,
                deepcopy(started["invalidated_on_success"])
                if started is not None
                else None,
            )
            self._next_id += 1
            self._by_id[execution.snapshot().execution] = execution
            if op is not None:
                self._by_op[op] = execution
                if start_worker:
                    execution.start()
            return execution

    def accept_start(
        self,
        execution: AnalysisExecution,
        started: dict[str, Any],
        *,
        start_worker: bool = True,
    ) -> None:
        """Bind a late native receipt to its registered completion owner.

        start_worker=False leaves observation to complete_in_current_worker;
        otherwise start the ordinary background worker. Conflicting operation
        ownership raises GuiRpcError. This never creates a second owner.
        """
        with self._lock:
            op = started["handle"]
            execution.observe_start(started)
            if op in self._by_op and self._by_op[op] is not execution:
                raise GuiRpcError(
                    "Analysis operation already has a completion owner",
                    reason="incompatible_wire",
                )
            self._by_op[op] = execution
            if start_worker:
                execution.start()

    def for_op(self, op: int) -> AnalysisExecution | None:
        """Find an existing completion owner without creating a new job."""
        with self._lock:
            return self._by_op.get(op)

    def get(self, execution: str) -> AnalysisExecution:
        """Resolve a session-local execution without binding or reconnecting."""
        with self._lock:
            found = self._by_id.get(execution)
        if found is None:
            raise GuiRpcError(
                f"unknown execution: {execution!r}", reason="unknown_execution"
            )
        return found

    def snapshots(self) -> list[ExecutionSnapshot]:
        """Return detached snapshots of every execution in this session."""
        with self._lock:
            executions = list(self._by_id.values())
        return [execution.snapshot() for execution in executions]

    def stop_admission(self) -> None:
        """Permanently reject new workers and wake existing ones."""
        self._closed.set()
        with self._lock:
            for execution in self._by_id.values():
                execution.wake()

    def join(self) -> None:
        """Join admitted workers after transport disconnect, before PNG cleanup."""
        with self._lock:
            executions = list(self._by_id.values())
        for execution in executions:
            execution.join()
