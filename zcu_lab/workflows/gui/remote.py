"""App-owned wire projection onto the existing T1 Window controls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

from pydantic import JsonValue
from zcu_tools.experiment.workflows import (
    Actor,
    InvalidRunState,
    RunMismatch,
    RunStatus,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import FormErrors, FormWritePath
from zcu_tools.gui.remote.control_service import (
    ControlOptions,
    RemoteControlServiceBase,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.method_spec import BoundMethod
from zcu_tools.gui.session.ports import OwnerScheduler

from .remote_methods import GUI_VERSION, METHOD_SPECS, WIRE_VERSION

if TYPE_CHECKING:
    from .window import T1WorkflowWindow


class RunTarget(TypedDict):
    """run_id is the non-empty native identity supplied to Start, never a token."""

    run_id: str


class FormWrite(TypedDict):
    """One JSON batch edit: nonempty string field path and JSON value.

    path uses model-field segments, not dotted keys, mapping keys or indexes.
    value may be an ordinary value, expression or complex tag. Native FormState
    owns branch ordering, whole-leaf replacement and field/model validation.
    """

    path: list[str]
    value: JsonValue


class StartParams(RunTarget):
    """Start identity and two required partial-input batches.

    plan/tunables contain FormWrite objects; [] uses current captured defaults.
    Each path occurs once. They never edit the existing GUI draft before success.
    """

    plan: list[FormWrite]
    tunables: list[FormWrite]


class TunablesParams(RunTarget):
    """External edit of one native run using its current complete parsed input.

    batch contains unique FormWrite paths. [] submits current input, not defaults.
    Actor is app-owned agent/rpc; caller supplies no revision or holder.
    """

    batch: list[FormWrite]


class FormErrorReply(TypedDict):
    """Existing FormErrors projection: relative field/index path and native message.

    Empty path means a model-wide error. No code, identity or other metadata
    is inferred from the native message.
    """

    path: list[str | int]
    message: str


class ProgressReply(TypedDict):
    """Native display snapshot: name is the bar; total may be null; completed is absolute."""

    name: str
    total: float | None
    completed: float


class NativeStatusReply(TypedDict):
    """JSON projection of RunStatus, with no inferred acceptance or completion.

    run_id/workflow identify the native run and declaration. lifecycle is running,
    pausing, paused, stopping, done, aborted, stopped or failed. call_seq is the
    last invocation or null; committed_seq counts successful Next commits.
    revision is the native numeric tunables version. reason is a native diagnostic
    or null. progress is the list of native display snapshots, not commit proof.
    """

    run_id: str
    workflow: str
    lifecycle: str
    call_seq: int | None
    committed_seq: int
    revision: int
    reason: str | None
    progress: list[ProgressReply]


class WorkflowReply(TypedDict):
    """Same-owner receipt using only existing Controller and Engine fields.

    run_id is the attached Start identity, or null before submission.
    operation_id is the current pending/executing segment token, or null.
    status is native RunStatus projected to JSON, or null before acceptance.
    A non-null operation_id never establishes worker acceptance or completion.
    """

    run_id: str | None
    operation_id: int | None
    status: NativeStatusReply | None


class T1RemoteControlAdapter(RemoteControlServiceBase):
    """RPC face of one T1 Window; it never owns a second run or capture.

    workflow.start submits partial path/value batches through Window.start;
    FormErrors becomes invalid_params/form_errors; data is an object whose native
    messages field contains the list of FormErrorReply path/message entries.
    Batch wire admission rejects malformed or duplicate paths without side effects.
    A successful reply is pending admission, not worker acceptance. GUI drafts are
    untouched on failure; success resets their bases through the existing owner.
    workflow.tunables requires run_id and a path/value batch. It submits complete
    current parsed draft through the same Window tuning panel with agent/rpc actor;
    [] never resets defaults. Success returns WorkflowReply with native revision.
    Invalid input stays in draft and projects FormErrors; accepted input/base and
    native values/revision are unchanged. Pending startup is not editable.
    workflow.status reads the same Controller on the owner. Pause/Resume/Stop
    require explicit run_id and invoke the existing Window control once.
    RunMismatch becomes precondition_failed/run_mismatch with expected/actual;
    InvalidRunState becomes precondition_failed/invalid_run_state with action/
    lifecycle. Unmarked pending RuntimeError and unexpected errors retain shared
    controller_error classification. Transport timeout/disconnect does not Stop
    a run and gives no retry guarantee. There are no business event projections,
    additional observation policies, holder checks or hidden pre-reads.
    """

    ctrl: T1WorkflowWindow

    def __init__(
        self,
        window: T1WorkflowWindow,
        opts: ControlOptions,
        *,
        bus: BaseEventBus,
        owner_scheduler: OwnerScheduler,
    ) -> None:
        """Bind one live Window, its session bus and its owner scheduler.

        Construct on the Qt owner after window exists. opts controls transport;
        constructor opens no socket. Caller starts/stops this adapter through
        inherited start()/stop(), stops it before destroying window, and must use
        the same session bus/owner as window. Stopping transport does not cancel
        native work. Registry drift and shared transport admission failures
        propagate without retry; no hardware is contacted by construction.
        """
        self._event_bus = bus
        super().__init__(
            window,
            opts,
            owner_scheduler=owner_scheduler,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="T1WorkflowRemoteServer",
            method_registry={
                "workflow.start": BoundMethod(
                    self._start, METHOD_SPECS["workflow.start"]
                ),
                "workflow.tunables": BoundMethod(
                    self._tunables, METHOD_SPECS["workflow.tunables"]
                ),
                "workflow.status": BoundMethod(
                    self._status, METHOD_SPECS["workflow.status"]
                ),
                "workflow.pause": BoundMethod(
                    self._pause, METHOD_SPECS["workflow.pause"]
                ),
                "workflow.resume": BoundMethod(
                    self._resume, METHOD_SPECS["workflow.resume"]
                ),
                "workflow.stop": BoundMethod(self._stop, METHOD_SPECS["workflow.stop"]),
            },
            event_serializers={},
            wire_event_name=str,
        )

    def _get_bus(self) -> BaseEventBus:
        return self._event_bus

    def _start(
        self,
        _adapter: RemoteControlServiceBase,
        params: StartParams,
    ) -> WorkflowReply:
        plan_batch = _parse_start_batch(params["plan"])
        tunables_batch = _parse_start_batch(params["tunables"])
        try:
            result = self.ctrl.start(
                params["run_id"],
                plan_batch=plan_batch,
                tunables_batch=tunables_batch,
            )
        except InvalidRunState as exc:
            raise _project_control_error(exc) from exc
        if isinstance(result, FormErrors):
            messages: list[FormErrorReply] = [
                {"path": list(path), "message": message}
                for path, message in result.messages.items()
            ]
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "Invalid Start form input",
                reason="form_errors",
                data={"messages": messages},
            )
        return self._status(_adapter, {})

    def _tunables(
        self,
        _adapter: RemoteControlServiceBase,
        params: TunablesParams,
    ) -> WorkflowReply:
        result = self.ctrl.apply_tunables(
            params["run_id"],
            _parse_start_batch(params["batch"]),
            actor=Actor("agent", "rpc"),
        )
        del result
        raise NotImplementedError(
            "Project FormErrors and native controls without retry"
        )

    def _status(
        self,
        _adapter: RemoteControlServiceBase,
        _params: Mapping[str, object],
    ) -> WorkflowReply:
        controller = self.ctrl.controller
        if controller is None:
            return {"run_id": None, "operation_id": None, "status": None}
        return {
            "run_id": controller.run_id,
            "operation_id": controller.operation_id,
            "status": _project_status(controller.status()),
        }

    def _pause(
        self,
        _adapter: RemoteControlServiceBase,
        params: RunTarget,
    ) -> WorkflowReply:
        try:
            status = self.ctrl.pause(params["run_id"])
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        controller = self.ctrl.controller
        return {
            "run_id": None if controller is None else controller.run_id,
            "operation_id": None if controller is None else controller.operation_id,
            "status": _project_status(status),
        }

    def _resume(
        self,
        _adapter: RemoteControlServiceBase,
        params: RunTarget,
    ) -> WorkflowReply:
        try:
            operation_id = self.ctrl.resume(params["run_id"])
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        controller = self.ctrl.controller
        return {
            "run_id": None if controller is None else controller.run_id,
            "operation_id": operation_id,
            "status": _project_status(
                None if controller is None else controller.status()
            ),
        }

    def _stop(
        self,
        _adapter: RemoteControlServiceBase,
        params: RunTarget,
    ) -> WorkflowReply:
        try:
            status = self.ctrl.stop(params["run_id"])
        except (RunMismatch, InvalidRunState) as exc:
            raise _project_control_error(exc) from exc
        controller = self.ctrl.controller
        return {
            "run_id": None if controller is None else controller.run_id,
            "operation_id": None if controller is None else controller.operation_id,
            "status": _project_status(status),
        }


def _parse_start_batch(
    batch: JsonValue | list[FormWrite],
) -> dict[FormWritePath, JsonValue]:
    """Convert the JSON batch shape only; FormState owns values and field paths."""
    if not isinstance(batch, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "Start batch must be an array")
    writes: dict[FormWritePath, JsonValue] = {}
    for edit in batch:
        if not isinstance(edit, dict) or set(edit) != {"path", "value"}:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "Start edits require path and value"
            )
        path = edit["path"]
        if not isinstance(path, list) or not path:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "Start paths must be nonempty arrays"
            )
        segments: list[str] = []
        for segment in path:
            if not isinstance(segment, str) or not segment:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    "Start path segments must be nonempty strings",
                )
            segments.append(segment)
        field_path = tuple(segments)
        if field_path in writes:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "Duplicate Start path")
        writes[field_path] = edit["value"]
    return writes


def _project_status(status: RunStatus | None) -> NativeStatusReply | None:
    """Project only native status fields; None preserves pending acceptance."""
    if status is None:
        return None
    return {
        "run_id": status.run_id,
        "workflow": status.workflow,
        "lifecycle": status.lifecycle,
        "call_seq": status.call_seq,
        "committed_seq": status.committed_seq,
        "revision": status.revision,
        "reason": status.reason,
        "progress": [
            {
                "name": progress.name,
                "total": progress.total,
                "completed": progress.completed,
            }
            for progress in status.progress
        ],
    }


def _project_control_error(exc: RunMismatch | InvalidRunState) -> RemoteError:
    """Translate native target/lifecycle failures without classifying other errors."""
    if isinstance(exc, RunMismatch):
        return RemoteError(
            ErrorCode.PRECONDITION_FAILED,
            str(exc),
            reason="run_mismatch",
            data={"expected": exc.expected, "actual": exc.actual},
        )
    return RemoteError(
        ErrorCode.PRECONDITION_FAILED,
        str(exc),
        reason="invalid_run_state",
        data={"action": exc.action, "lifecycle": exc.lifecycle},
    )
