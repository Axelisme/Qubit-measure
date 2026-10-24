"""Explicit workflow record assembly uses delivered operation facts, not live state."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from zcu_tools.experiment import workflows
from zcu_tools.experiment.acquisition_facts import AcquisitionFacts
from zcu_tools.experiment.metadata import RunMetadata
from zcu_tools.experiment.native_persistence import load_run_with_metadata, save_run
from zcu_tools.experiment.records import RunRecord

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore


def test_historical_documents_default_to_unrecorded_acquisitions() -> None:
    metadata = native_metadata().model_dump_json(exclude={"acquisitions"})
    assert RunMetadata.model_validate_json(metadata).acquisitions == ()
    operation = TypeAdapter(workflows.OperationFacts).validate_json(
        '{"kind":"run","iteration":1,"attempt":1,'
        '"started_at":"2026-10-22T00:00:00Z","finished_at":"2026-10-22T00:00:01Z",'
        '"device_snapshot":{},"outcome":"completed","reason":null,"stopped":false}'
    )
    assert operation.name is None
    assert operation.acquisitions == ()


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
        acquisitions=(AcquisitionFacts(8, 2, 2, True, 12.5),),
    )
    record = workflows.record_with_metadata(
        workflows.Completed(source, partial_result=True, facts=facts),
        identity=workflows.RunIdentity(
            "0123456789abcdef0123456789abcdef", history.provenance
        ),
        experiment_tag=core.AXES_SPEC.tag,
        snapshot=history.snapshot,
    )
    assert record.metadata is not None
    assert record.metadata.completion == "stopped"
    assert record.metadata.acquisitions == facts.acquisitions
    path = tmp_path / "descendant.h5"
    save_run(record, path, spec=core.AXES_SPEC)
    restored = load_run_with_metadata(path, spec=core.AXES_SPEC)
    assert restored.metadata == record.metadata
    assert restored.metadata is not None
    assert restored.metadata.run_id == "0123456789abcdef0123456789abcdef-000003-7"
