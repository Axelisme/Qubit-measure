"""Native workflow facts projected without lifecycle or Form policy."""

from typing import Literal, TypedDict

from pydantic import JsonValue

from zcu_tools.datafile import JsonObject
from zcu_tools.experiment.workflows import CommittedRecord, OperationFact, RunStatus
from zcu_tools.experiment.workflows.models import OperationKind, OperationOutcome


class EncodedRecordReply(TypedDict):
    """Native encoding: json/repr mode, JSON/diagnostic value, optional failure."""

    mode: Literal["json", "repr"]
    value: JsonValue
    error: str | None


class OperationReply(TypedDict):
    """One native attempt; UTC timestamps are ISO text, snapshot is JSON or None.

    kind identifies the operation helper; attempt counts actual executor starts
    within this commit scope. outcome/reason retain native delivery classification.
    """

    kind: OperationKind
    attempt: int
    started_at: str
    finished_at: str
    device_snapshot: dict[str, JsonObject] | None
    outcome: OperationOutcome
    reason: str | None


class RecordReply(TypedDict):
    """One native committed point, with no inferred values or artist inspection.

    commit_seq counts records; call_seq identifies the commit scope; revision is
    its initial tunables version and revisions lists every version read in order.
    encoded_record is the native record. directory is relative to data_root.
    committed_at is UTC ISO text; record_type names the module-qualified model. Workflow file references belong to
    encoded_record, whose contents the framework does not interpret.
    """

    commit_seq: int
    call_seq: int
    revision: int
    encoded_record: EncodedRecordReply
    directory: str
    committed_at: str
    revisions: list[int]
    record_type: str
    operations: list[OperationReply]


class ProgressReply(TypedDict):
    """Native display snapshot: name is the bar; total may be null; completed is absolute."""

    name: str
    total: float | None
    completed: float


class NativeStatusReply(TypedDict):
    """JSON projection of RunStatus, with no inferred acceptance or completion.

    run_id/workflow identify the native run and declaration. lifecycle is running,
    pausing, paused, stopping, done, cancelled or failed. call_seq is the
    current commit scope or null; committed_seq counts successful commits.
    revision is the native numeric tunables version. reason is a native diagnostic
    or null. progress is the list of native display snapshots, not commit proof.
    pending_request_id names the exact answerable Ask, or null outside Ask.
    """

    run_id: str
    workflow: str
    lifecycle: str
    call_seq: int | None
    committed_seq: int
    revision: int
    reason: str | None
    progress: list[ProgressReply]
    pending_request_id: str | None


def project_record(record: CommittedRecord) -> RecordReply:
    """Project native fields only; keep the detached encoded value unchanged."""
    return {
        "commit_seq": record.commit_seq,
        "call_seq": record.call_seq,
        "revision": record.revision,
        "encoded_record": {
            "mode": record.encoded_record.mode,
            "value": record.encoded_record.value,
            "error": record.encoded_record.error,
        },
        "directory": record.directory,
        "committed_at": record.committed_at.isoformat(),
        "revisions": list(record.revisions),
        "record_type": record.record_type,
        "operations": [project_operation(operation) for operation in record.operations],
    }


def project_operation(operation: OperationFact) -> OperationReply:
    """Project one detached native attempt without reading devices or files."""
    return {
        "kind": operation.kind,
        "attempt": operation.attempt,
        "started_at": operation.started_at.isoformat(),
        "finished_at": operation.finished_at.isoformat(),
        "device_snapshot": None
        if operation.device_snapshot is None
        else dict(operation.device_snapshot),
        "outcome": operation.outcome,
        "reason": operation.reason,
    }


def project_status(status: RunStatus | None) -> NativeStatusReply | None:
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
        "pending_request_id": None
        if status.pending is None
        else status.pending.request_id,
        "progress": [
            {
                "name": progress.name,
                "total": progress.total,
                "completed": progress.completed,
            }
            for progress in status.progress
        ],
    }
