"""Typed payloads for the engine-owned journal, not a second host API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from traceback import TracebackException
from typing import Literal

from pydantic import JsonValue

from .models import Actor, Capability, CommittedRecord, RunIdentity, RunPaths
from .ports import DeviceSnapshot


@dataclass(frozen=True)
class RunMetadata:
    """Detached startup provenance used for the manifest and first journal line.

    identity is host-supplied provenance. workflow is the declared name. plan
    and tunables are complete, validated JSON model projections at revision 0.
    requires lists declared capabilities. roots are absolute run directories.
    started_at is the original timezone-aware UTC start time.
    """

    identity: RunIdentity
    workflow: str
    plan: JsonValue
    tunables: JsonValue
    requires: tuple[Capability, ...]
    roots: RunPaths
    started_at: datetime


@dataclass(frozen=True)
class RunStarted(RunMetadata):
    """Startup metadata; kind is run_started and revision is always zero."""

    kind: Literal["run_started"] = field(default="run_started", init=False)
    revision: Literal[0] = field(default=0, init=False)


@dataclass(frozen=True)
class IterationStarted:
    """Invocation provenance before workflow code runs.

    call_seq counts step calls from 1. commit_seq is the number already
    committed, initially zero. revision is captured for this call.
    iteration_dir is the data-root-relative iter/<call_seq> directory.
    """

    call_seq: int
    commit_seq: int
    revision: int
    iteration_dir: str
    kind: Literal["iteration_started"] = field(default="iteration_started", init=False)


@dataclass(frozen=True)
class ErrorInfo:
    """Original error diagnostics: qualified type name, message, and traceback."""

    type: str
    message: str
    traceback: str


def describe_error(error: BaseException) -> ErrorInfo:
    """Capture the original error and chained traceback without replacing it.

    error is the cause selected by the engine's failure policy. Diagnostic
    formatting failures propagate; they cannot be treated as record degradation.
    """
    error_type = type(error)
    return ErrorInfo(
        type=f"{error_type.__module__}.{error_type.__qualname__}",
        message=str(error),
        traceback="".join(TracebackException.from_exception(error).format()),
    )


@dataclass(frozen=True)
class ExperimentFailed:
    """Recoverable Schedule failure without a Result or run file.

    call_seq and run_no identify the invocation and effect, each counted from 1.
    experiment is the validated function-name stem. reason is the Schedule
    diagnostic; error preserves the original cause.
    """

    call_seq: int
    run_no: int
    experiment: str
    reason: str
    error: ErrorInfo
    kind: Literal["experiment_failed"] = field(default="experiment_failed", init=False)


@dataclass(frozen=True)
class StepCommitted(CommittedRecord):
    """Journal-authoritative Next record; inherited fields retain their meaning."""

    kind: Literal["step_committed"] = field(default="step_committed", init=False)


@dataclass(frozen=True)
class IterationResult:
    """Invocation output reference shared by non-commit outcomes.

    call_seq counts calls from 1; revision is the call's captured tunables
    revision. iteration_dir is relative to data_root. run_files lists only
    published finals relative to that iteration, ordered by run_no.
    """

    call_seq: int
    revision: int
    iteration_dir: str
    run_files: tuple[str, ...]


@dataclass(frozen=True)
class StepFinished(IterationResult):
    """Terminal workflow return: outcome is done/aborted, reason is diagnostic.

    done uses reason=None; aborted uses the workflow's nonempty reason.
    Neither outcome creates a point record.
    """

    outcome: Literal["done", "aborted"]
    reason: str | None
    kind: Literal["step_finished"] = field(default="step_finished", init=False)


@dataclass(frozen=True)
class StepDiscarded(IterationResult):
    """Cancelled invocation; reason is pause or stop, not a failure cause."""

    reason: Literal["pause", "stop"]
    kind: Literal["step_discarded"] = field(default="step_discarded", init=False)


@dataclass(frozen=True)
class StepFailed(IterationResult):
    """Failed invocation; error preserves the engine-selected original cause."""

    error: ErrorInfo
    kind: Literal["step_failed"] = field(default="step_failed", init=False)


@dataclass(frozen=True)
class ChangedValue:
    """One applied leaf change: dot path and detached old/new JSON values."""

    path: str
    old: JsonValue
    new: JsonValue


@dataclass(frozen=True)
class TunablesChanged:
    """One validated tunables batch attributed to actor.

    revision_before/after identify consecutive versions. changes records the
    applied values in request order, not unvalidated caller input.
    """

    actor: Actor
    revision_before: int
    revision_after: int
    changes: tuple[ChangedValue, ...]
    kind: Literal["tunables_changed"] = field(default="tunables_changed", init=False)


@dataclass(frozen=True)
class ControlRequest:
    """Requested control time and current invocation, or None before any call.

    requested_at is timezone-aware UTC. call_seq counts invocations from 1.
    """

    requested_at: datetime
    call_seq: int | None


@dataclass(frozen=True)
class PauseRequested(ControlRequest):
    """Pause request; kind is pause_requested, not proof of producer shutdown."""

    kind: Literal["pause_requested"] = field(default="pause_requested", init=False)


@dataclass(frozen=True)
class StopRequested(ControlRequest):
    """Stop request; kind is stop_requested, not proof of producer shutdown."""

    kind: Literal["stop_requested"] = field(default="stop_requested", init=False)


@dataclass(frozen=True)
class Paused:
    """Producer has stopped; call_seq may be None, commit_seq counts commits."""

    call_seq: int | None
    commit_seq: int
    kind: Literal["paused"] = field(default="paused", init=False)


@dataclass(frozen=True)
class Resumed:
    """Segment resume provenance after the host acquired its device lease.

    commit_seq counts existing commits; revision is the next call's version.
    devices is the detached host-supplied current device snapshot.
    """

    commit_seq: int
    revision: int
    devices: DeviceSnapshot
    kind: Literal["resumed"] = field(default="resumed", init=False)


@dataclass(frozen=True)
class RunEnded:
    """Terminal summary without a state snapshot.

    lifecycle is done/aborted/stopped/failed. reason is diagnostic or None.
    call_seq is None before any invocation; commit_seq counts committed Nexts.
    """

    lifecycle: Literal["done", "aborted", "stopped", "failed"]
    reason: str | None
    call_seq: int | None
    commit_seq: int
    kind: Literal["run_ended"] = field(default="run_ended", init=False)


type JournalPayload = (
    RunStarted
    | IterationStarted
    | ExperimentFailed
    | StepCommitted
    | StepFinished
    | StepDiscarded
    | StepFailed
    | TunablesChanged
    | PauseRequested
    | Paused
    | Resumed
    | StopRequested
    | RunEnded
)
