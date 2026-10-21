"""Native workflow facts projected without lifecycle or Form policy."""

from typing import Literal, TypedDict

from pydantic import JsonValue

from zcu_tools.experiment.workflows import CommittedRecord, RunStatus


class EncodedRecordReply(TypedDict):
    """Native encoding: json/repr mode, JSON/diagnostic value, optional failure."""

    mode: Literal["json", "repr"]
    value: JsonValue
    error: str | None


class RecordReply(TypedDict):
    """One native committed point, with no inferred values or artist inspection.

    commit_seq counts Next commits; call_seq counts invocations; revision is the
    invocation's captured tunables version. encoded_record is the native summary.
    iteration_dir is relative to data_root. Workflow file references belong to
    encoded_record, whose contents the framework does not interpret.
    """

    commit_seq: int
    call_seq: int
    revision: int
    encoded_record: EncodedRecordReply
    iteration_dir: str


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
        "iteration_dir": record.iteration_dir,
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
        "progress": [
            {
                "name": progress.name,
                "total": progress.total,
                "completed": progress.completed,
            }
            for progress in status.progress
        ],
    }
