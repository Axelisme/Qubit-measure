"""Explicit workflow record assembly uses delivered operation facts, not live state."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from zcu_tools.experiment import workflows
from zcu_tools.experiment.native_persistence import load_run_with_metadata, save_run
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
        identity=workflows.RunIdentity(
            "0123456789abcdef0123456789abcdef", history.provenance
        ),
        experiment_tag="synthetic/trace",
        snapshot=history.snapshot,
    )
    assert record.result is source.result
    assert record.metadata is not None
    assert record.metadata.run_id == "0123456789abcdef0123456789abcdef-000003-7"
    assert record.metadata.experiment == "synthetic/trace"
    assert record.metadata.snapshot == history.snapshot
    assert record.metadata.started_at == "2026-10-22T01:02:03+00:00"
    assert record.metadata.finished_at == "2026-10-22T01:02:05+00:00"
    assert record.metadata.completion == ("stopped" if stopped else "complete")
    assert record.metadata.provenance == history.provenance
    assert record.metadata.device_snapshot == {"flux": {"value": 0.125}}
    assert source.metadata is None


def test_descendant_cancellation_is_saved_as_stopped(tmp_path: Path) -> None:
    history = native_metadata()
    core = SavingCore()
    cfg = SavingCfg()
    source = RunRecord(cfg, core.run(cfg, context=None))
    facts = workflows.OperationFacts(
        kind="run",
        iteration=3,
        attempt=7,
        started_at=datetime(2026, 10, 22, 1, 2, 3, tzinfo=UTC),
        finished_at=datetime(2026, 10, 22, 1, 2, 5, tzinfo=UTC),
        device_snapshot={},
        outcome="completed",
        reason=None,
        stopped=False,
    )
    record = workflows.record_with_metadata(
        workflows.Completed(source, partial_result=True, facts=facts),
        identity=workflows.RunIdentity(
            "0123456789abcdef0123456789abcdef", history.provenance
        ),
        experiment_tag=core.persistence_spec().tag,
        snapshot=history.snapshot,
    )
    assert record.metadata is not None
    assert record.metadata.completion == "stopped"
    path = tmp_path / "descendant.h5"
    save_run(record, path, spec=core.persistence_spec())
    restored = load_run_with_metadata(path, spec=core.persistence_spec())
    assert restored.metadata == record.metadata
    assert restored.metadata is not None
    assert restored.metadata.run_id == "0123456789abcdef0123456789abcdef-000003-7"
