"""Committed workflow wire records expose named executor operations."""

from dataclasses import replace
from datetime import UTC, datetime

from zcu_tools.experiment.workflows import (
    CommittedRecord,
    EncodedRecord,
    OperationFacts,
)
from zcu_tools.gui.app.measure.remote.workflow_types import project_record


def test_record_reply_preserves_executor_operations_and_names() -> None:
    timestamp = datetime(2026, 10, 25, tzinfo=UTC)
    operation = OperationFacts(
        kind="run",
        iteration=1,
        attempt=1,
        started_at=timestamp,
        finished_at=timestamp,
        device_snapshot={"bias": {"value": 0.125}},
        outcome="completed",
        reason=None,
        stopped=False,
        name="Synthetic trace",
    )
    record = CommittedRecord(
        commit_seq=1,
        call_seq=1,
        revision=0,
        encoded_record=EncodedRecord("json", {"value": 3}, None),
        directory="iteration-000001",
        committed_at=timestamp,
        revisions=(0,),
        record_type="synthetic.Record",
        operations=(operation, replace(operation, attempt=2, name=None)),
    )

    reply = project_record(record)

    expected = {
        "kind": "run",
        "iteration": 1,
        "attempt": 1,
        "started_at": "2026-10-25T00:00:00+00:00",
        "finished_at": "2026-10-25T00:00:00+00:00",
        "device_snapshot": {"bias": {"value": 0.125}},
        "outcome": "completed",
        "reason": None,
        "stopped": False,
        "name": "Synthetic trace",
    }
    assert reply["operations"] == [
        expected,
        {**expected, "attempt": 2, "name": None},
    ]
    assert reply["encoded_record"] == {
        "mode": "json",
        "value": {"value": 3},
        "error": None,
    }
