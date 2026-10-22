"""Typed workflow outcomes, controls, records and host-supplied provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from zcu_tools.datafile import JsonObject, SoftwareProvenance
from zcu_tools.experiment.script_host import PendingQuestion

type Lifecycle = Literal[
    "running", "pausing", "paused", "stopping", "done", "cancelled", "failed"
]


@dataclass(frozen=True)
class Completed[T]:
    """A returned value, partial-completion flag and operation attempt facts.

    value is fn's payload; env.run uses the existing RunRecord with cfg/data.
    partial_result is True when the operation or a completed descendant was
    stopped, not a scientific completeness check or a persistence guarantee.
    facts is the required OperationFacts captured for this delivered attempt,
    also written to journal and the enclosing commit. Its stopped bit reports
    only the operation token's own stop state, independently of descendant stops.
    """

    value: T
    partial_result: bool
    facts: OperationFacts


@dataclass(frozen=True)
class Failed:
    """A recoverable acquire/analysis failure, without partial data.

    reason describes the cause. Authors decide whether to retry or continue;
    arbitrary framework/build/conversion exceptions instead propagate.
    """

    reason: str


@dataclass(frozen=True)
class Actor:
    """Editor attribution: kind is user/agent; name is a nonempty identity."""

    kind: Literal["user", "agent"]
    name: str

    def __post_init__(self) -> None:
        if self.kind not in ("user", "agent") or not self.name:
            raise ValueError("Actor requires kind user/agent and a nonempty name")


@dataclass(frozen=True)
class TunablesSnapshot:
    """Detached complete model JSON in values, at the nonnegative revision."""

    revision: int
    values: JsonValue


@dataclass(frozen=True)
class ProgressSnapshot:
    """Display-only bar: name, optional target total, absolute completed value."""

    name: str
    total: float | None
    completed: float


@dataclass(frozen=True)
class RunStatus:
    """Detached lifecycle, progress and committed-record counters.

    run_id/workflow identify the run. lifecycle is a Lifecycle value. call_seq
    identifies the current commit scope, or None before execute. committed_seq
    counts published records, independent of allocated directory numbers.
    revision is the latest tunables version. reason describes failure or None.
    progress is display-only. pending holds the exact request id, authored
    question and declared answer type while an Ask is answerable, otherwise None.
    """

    run_id: str
    workflow: str
    lifecycle: Lifecycle
    call_seq: int | None
    committed_seq: int
    revision: int
    reason: str | None
    progress: tuple[ProgressSnapshot, ...]
    pending: PendingQuestion | None = None


@dataclass(frozen=True)
class RunPaths:
    """Distinct new run roots: metadata_root for ledger; data_root for commits.

    Start refuses existing/overlapping roots. Authors own files in allocated
    commit directories. The framework never scans or interprets native files.
    """

    metadata_root: Path
    data_root: Path


@dataclass(frozen=True)
class RunIdentity:
    """Stable run_id, native software provenance and optional entry identity.

    entry_id/entry_label are host-supplied workpoint evidence or None. Unknown
    provenance remains None in SoftwareProvenance; it is not inferred here.
    """

    run_id: str
    provenance: SoftwareProvenance
    entry_id: str | None = None
    entry_label: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("RunIdentity.run_id must not be empty")


@dataclass(frozen=True)
class EncodedRecord:
    """Detached record projection: JSON value or diagnostic repr degradation.

    mode is json/repr. value is JSON data or a diagnostic string. error is None
    for successful encoding and describes serialization/repr failure otherwise.
    """

    mode: Literal["json", "repr"]
    value: JsonValue
    error: str | None


type OperationKind = Literal["run", "call", "setup_device", "wait_until"]
type OperationOutcome = Literal["completed", "partial", "discarded", "failed", "raised"]


@dataclass(frozen=True)
class OperationFacts:
    """One actual executor attempt, retained even when Pause discards its value.

    kind names run/call/setup_device/wait_until. iteration starts at one and
    advances only after successful commit, not directory allocation. attempt
    counts from one within that scope, including retries. started_at/finished_at are UTC
    worker boundaries. device_snapshot is detached pre-call driver JSON for run,
    otherwise None. outcome is completed, partial, discarded by Pause, failed
    acquire, or raised exception. reason is a diagnostic or None on completion.
    stopped is the operation token's completion-time stop bit, including failed,
    raised and discarded attempts; it is not inferred from later run state.
    """

    kind: OperationKind
    iteration: int
    attempt: int
    started_at: datetime
    finished_at: datetime
    device_snapshot: Mapping[str, JsonObject] | None
    outcome: OperationOutcome
    reason: str | None
    stopped: bool


@dataclass(frozen=True)
class CommittedRecord:
    """One journal-authoritative record, independent of directory allocation.

    commit_seq counts successful commits; call_seq identifies the commit scope.
    revision is the scope's initial tunables version. revisions lists versions
    actually read, deduplicated in read order. encoded_record is detached JSON.
    directory is relative to data_root, e.g. commits/000007. committed_at is UTC.
    record_type is the model's module-qualified name, keyed in manifest schemas.
    operations retains this scope's actual executor attempts in start order.
    """

    commit_seq: int
    call_seq: int
    revision: int
    encoded_record: EncodedRecord
    directory: str
    committed_at: datetime
    revisions: tuple[int, ...]
    record_type: str
    operations: tuple[OperationFacts, ...] = ()


class QuestionMismatch(ValueError):
    """A rejected stale answer; pending_request_id names the current request or None."""

    def __init__(self, request_id: str, pending_request_id: str | None) -> None:
        self.request_id = request_id
        self.pending_request_id = pending_request_id
        super().__init__(
            f"Question {request_id!r} is not pending; current request is {pending_request_id!r}"
        )


class MissingCapability(RuntimeError):
    """A required capability is absent; capability identifies the missing port."""

    def __init__(self, capability: str) -> None:
        self.capability = capability
        super().__init__(f"Missing workflow capability: {capability}")


class RunMismatch(ValueError):
    """Requested expected run identity differs from attached actual identity."""

    def __init__(self, expected: str, actual: str | None) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Requested run {expected!r}, attached run is {actual!r}")


class InvalidRunState(RuntimeError):
    """Illegal action for the current lifecycle; both fields describe rejection."""

    def __init__(self, action: str, lifecycle: Lifecycle) -> None:
        self.action = action
        self.lifecycle = lifecycle
        super().__init__(f"Cannot {action} while workflow is {lifecycle}")


class RevisionConflict(ValueError):
    """Stale expected tunables version; actual is the current published revision."""

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"Expected tunables revision {expected}, actual is {actual}")
