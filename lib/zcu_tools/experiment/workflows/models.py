"""Typed workflow outcomes, controls, records and host-supplied provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from zcu_tools.experiment.operations import Completed as Completed
from zcu_tools.experiment.operations import Failed as Failed
from zcu_tools.experiment.operations import MissingCapability as MissingCapability
from zcu_tools.experiment.operations import OperationFacts as OperationFacts
from zcu_tools.experiment.operations import OperationKind as OperationKind
from zcu_tools.experiment.operations import OperationOutcome as OperationOutcome
from zcu_tools.experiment.operations import RunIdentity as RunIdentity
from zcu_tools.experiment.script_host import PendingQuestion

type Lifecycle = Literal[
    "running", "pausing", "paused", "stopping", "done", "cancelled", "failed"
]


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
    active_path is the root-to-first-fork path through the live scope tree.
    live_paths lists every live leaf's full root-to-leaf scope path in child
    birth order, including the root itself when it has no children. It defaults
    to an empty list, never None.
    Both use slash-separated kind:name#seq segments, omitting :name when unnamed
    and #seq for the root. Author children use their label in place of kind.
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
    active_path: str = "script"
    live_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunPaths:
    """Distinct new run roots: metadata_root for ledger; data_root for commits.

    Start refuses existing/overlapping roots. Authors own files in allocated
    commit directories. The framework never scans or interprets native files.
    """

    metadata_root: Path
    data_root: Path


@dataclass(frozen=True)
class EncodedRecord:
    """Detached record projection: JSON value or diagnostic repr degradation.

    mode is json/repr. value is JSON data or a diagnostic string. error is None
    for successful encoding and describes serialization/repr failure otherwise.
    """

    mode: Literal["json", "repr"]
    value: JsonValue
    error: str | None


@dataclass(frozen=True)
class CommittedRecord:
    """One journal-authoritative record, independent of directory allocation.

    commit_seq counts successful commits; call_seq is the nearest iteration birth
    number, or zero outside iterations.
    revision is the scope's initial tunables version. revisions lists versions
    actually read, deduplicated in read order. encoded_record is detached JSON.
    directory is relative to data_root, e.g. iteration-000007/calibration-000001. committed_at is UTC.
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
