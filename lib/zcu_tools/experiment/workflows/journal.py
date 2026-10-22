"""Named payloads appended by the workflow execution/storage owner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from traceback import TracebackException
from typing import Literal

from pydantic import JsonValue, SerializeAsAny

from zcu_tools.device import BaseDeviceInfo

from .models import Actor, CommittedRecord, OperationFact, RunIdentity, RunPaths


@dataclass(frozen=True)
class RunMetadata:
    """Startup identity, declaration, detached plan/tunables, roots and UTC start."""

    identity: RunIdentity
    workflow: str
    plan: JsonValue
    tunables: JsonValue
    roots: RunPaths
    started_at: datetime


@dataclass(frozen=True)
class RunStarted(RunMetadata):
    """Revision-zero startup event; kind is run_started and revision is zero."""

    kind: Literal["run_started"] = field(default="run_started", init=False)
    revision: Literal[0] = field(default=0, init=False)


@dataclass(frozen=True)
class RecordCommitted(CommittedRecord):
    """CommittedRecord's journal representation, tagged record_committed."""

    kind: Literal["record_committed"] = field(default="record_committed", init=False)


@dataclass(frozen=True)
class OperationFinished:
    """Actual attempt evidence in commit scope call_seq, tagged operation_finished."""

    call_seq: int
    operation: OperationFact
    kind: Literal["operation_finished"] = field(
        default="operation_finished", init=False
    )


@dataclass(frozen=True)
class DeviceSetup:
    """Completed device invocation, with requested name and full typed cfg."""

    name: str
    cfg: SerializeAsAny[BaseDeviceInfo]
    kind: Literal["device_setup"] = field(default="device_setup", init=False)


@dataclass(frozen=True)
class ErrorInfo:
    """Original error evidence: qualified type, message and chained traceback."""

    type: str
    message: str
    traceback: str


def describe_error(error: BaseException) -> ErrorInfo:
    """Capture the original error and chain; diagnostic formatting errors raise."""
    error_type = type(error)
    return ErrorInfo(
        type=f"{error_type.__module__}.{error_type.__qualname__}",
        message=str(error),
        traceback="".join(TracebackException.from_exception(error).format()),
    )


@dataclass(frozen=True)
class ExperimentFailed:
    """Recoverable acquire failure without partial Result data.

    call_seq identifies the commit scope. run_no counts experiment requests in
    that scope. experiment names the callback. reason/error retain its cause.
    """

    call_seq: int
    run_no: int
    experiment: str
    reason: str
    error: ErrorInfo
    kind: Literal["experiment_failed"] = field(default="experiment_failed", init=False)


@dataclass(frozen=True)
class TunablesChanged:
    """Whole-model update: actor, consecutive revisions and complete new JSON."""

    actor: Actor
    revision_before: int
    revision_after: int
    tunables: dict[str, JsonValue]
    kind: Literal["tunables_changed"] = field(default="tunables_changed", init=False)


@dataclass(frozen=True)
class ControlRequest:
    """Control requested_at UTC, and current commit-scope call_seq or None."""

    requested_at: datetime
    call_seq: int | None


@dataclass(frozen=True)
class PauseRequested(ControlRequest):
    """Pause intent, tagged pause_requested; not proof of operation drain."""

    kind: Literal["pause_requested"] = field(default="pause_requested", init=False)


@dataclass(frozen=True)
class StopRequested(ControlRequest):
    """Explicit Stop intent, tagged stop_requested; author cleanup may still run."""

    kind: Literal["stop_requested"] = field(default="stop_requested", init=False)


@dataclass(frozen=True)
class Paused:
    """Reached suspension boundary: current call_seq and published commit_seq."""

    call_seq: int | None
    commit_seq: int
    kind: Literal["paused"] = field(default="paused", init=False)


@dataclass(frozen=True)
class Resumed:
    """Accepted continuation, retaining commit_seq and latest tunables revision."""

    commit_seq: int
    revision: int
    kind: Literal["resumed"] = field(default="resumed", init=False)


@dataclass(frozen=True)
class RunEnded:
    """Terminal lifecycle, reason, counters and summary-file presence.

    lifecycle is done/cancelled/failed. reason is optional diagnostic text.
    call_seq identifies the last commit scope or None; commit_seq counts records.
    summary is True only when a non-None done return was written to summary.json.
    """

    lifecycle: Literal["done", "cancelled", "failed"]
    reason: str | None
    call_seq: int | None
    commit_seq: int
    summary: bool = False
    kind: Literal["run_ended"] = field(default="run_ended", init=False)


type JournalPayload = (
    RunStarted
    | RecordCommitted
    | OperationFinished
    | DeviceSetup
    | ExperimentFailed
    | TunablesChanged
    | PauseRequested
    | StopRequested
    | Paused
    | Resumed
    | RunEnded
)
