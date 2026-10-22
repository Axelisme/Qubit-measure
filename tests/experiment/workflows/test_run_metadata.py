"""Explicit workflow record assembly uses delivered operation facts, not live state."""

from datetime import UTC, datetime

import pytest
from zcu_tools.experiment import workflows
from zcu_tools.experiment.records import RunRecord

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore


@pytest.mark.parametrize("stopped", [False, True])
def test_record_metadata_uses_workflow_identity_and_delivered_attempt(
    stopped: bool,
) -> None:
    history = native_metadata()
    cfg = SavingCfg()
    source = RunRecord(cfg, SavingCore().run(cfg, context=None))
    facts = workflows.OperationFacts(
        kind="run",
        iteration=3,
        attempt=7,
        started_at=datetime(2026, 10, 22, 1, 2, 3, tzinfo=UTC),
        finished_at=datetime(2026, 10, 22, 1, 2, 5, tzinfo=UTC),
        device_snapshot={"flux": {"value": 0.125}},
        outcome="partial" if stopped else "completed",
        reason=None,
        stopped=stopped,
    )
    completed = workflows.Completed(source, stopped, facts)
    record = workflows.record_with_metadata(
        completed,
        identity=workflows.RunIdentity(history.run_id, history.provenance),
        experiment_tag="synthetic/trace",
        snapshot=history.snapshot,
    )
    assert record.result is source.result
    assert record.metadata is not None
    assert record.metadata.run_id == "20261006T010000Z-a1b2c3-000003-7"
    assert record.metadata.experiment == "synthetic/trace"
    assert record.metadata.snapshot == history.snapshot
    assert record.metadata.started_at == "2026-10-22T01:02:03+00:00"
    assert record.metadata.finished_at == "2026-10-22T01:02:05+00:00"
    assert record.metadata.completion == ("stopped" if stopped else "complete")
    assert record.metadata.provenance == history.provenance
    assert record.metadata.device_snapshot == {"flux": {"value": 0.125}}
    assert source.metadata is None
