"""Committed workflow wire records retain named operations and acquisition facts."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from zcu_tools.experiment.acquisition_facts import AcquisitionFacts
from zcu_tools.experiment.workflows import (
    CommittedRecord,
    EncodedRecord,
    OperationFacts,
)
from zcu_tools.gui.app.measure.remote.workflow_types import (
    project_operation,
    project_record,
)


@pytest.mark.parametrize("seconds", [None, 0.0, 94.0])
def test_operation_reply_keeps_simulated_shot_time_and_omits_absence(
    seconds: float | None,
) -> None:
    timestamp = datetime(2026, 10, 25, tzinfo=UTC)
    operation = OperationFacts(
        kind="run",
        iteration=1,
        attempt=1,
        started_at=timestamp,
        finished_at=timestamp,
        device_snapshot=None,
        outcome="completed",
        reason=None,
        stopped=False,
        acquisitions=(
            AcquisitionFacts(10, 10, 1, False, simulated_physical_s=seconds),
        ),
    )
    acquisition = project_operation(operation)["acquisitions"][0]
    if seconds is None:
        assert "simulated_physical_s" not in acquisition
    else:
        assert acquisition.get("simulated_physical_s") == seconds


def test_record_reply_preserves_operation_names_and_ordered_acquisitions() -> None:
    timestamp = datetime(2026, 10, 25, tzinfo=UTC)
    operation = OperationFacts(
        kind="run",
        iteration=1,
        attempt=1,
        started_at=timestamp,
        finished_at=timestamp,
        device_snapshot=None,
        outcome="completed",
        reason=None,
        stopped=False,
        name="Synthetic trace",
        acquisitions=(
            AcquisitionFacts(2, 2, 1, False),
            AcquisitionFacts(8, 3, 2, True, 0.125),
            AcquisitionFacts(4, 1, 1, True),
            AcquisitionFacts(6, 0, 0, False),
        ),
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
        operations=(
            operation,
            replace(operation, attempt=2, name=None, acquisitions=()),
        ),
    )

    reply = project_record(record)

    assert [(op["name"], op["acquisitions"]) for op in reply["operations"]] == [
        (
            "Synthetic trace",
            [
                {
                    "requested_rounds": 2,
                    "completed_rounds": 2,
                    "attempts": 1,
                    "early_stopped": False,
                    "stop_value": None,
                },
                {
                    "requested_rounds": 8,
                    "completed_rounds": 3,
                    "attempts": 2,
                    "early_stopped": True,
                    "stop_value": 0.125,
                },
                {
                    "requested_rounds": 4,
                    "completed_rounds": 1,
                    "attempts": 1,
                    "early_stopped": True,
                    "stop_value": None,
                },
                {
                    "requested_rounds": 6,
                    "completed_rounds": 0,
                    "attempts": 0,
                    "early_stopped": False,
                    "stop_value": None,
                },
            ],
        ),
        (None, []),
    ]
