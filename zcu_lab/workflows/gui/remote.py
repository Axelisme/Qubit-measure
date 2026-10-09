"""App-owned wire projection onto the existing T1 Window controls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

from zcu_tools.experiment.workflows import InvalidRunState, RunMismatch, RunStatus
from zcu_tools.gui.event_bus import BaseEventBus
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
