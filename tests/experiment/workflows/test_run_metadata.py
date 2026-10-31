"""Explicit workflow metadata preserves core evidence and caller completion."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from zcu_tools import datafile
from zcu_tools.experiment import workflows
from zcu_tools.experiment.metadata import RunMetadataContent
from zcu_tools.experiment.native_persistence import save_run
from zcu_tools.experiment.records import RunRecord

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore


@pytest.mark.parametrize("completion", ["stopped", "complete"])
def test_explicit_completion_and_core_metadata_survive_native_save(
    tmp_path: Path, completion: Literal["stopped", "complete"]
) -> None:
    history = native_metadata()
    core = SavingCore()
    cfg = SavingCfg()
    source = RunRecord(
        cfg,
        core.run(cfg, context=None),
        metadata={"device_snapshot": {"flux": {"value": 0.125}}, "author": "kept"},
    )
    facts = workflows.OperationFacts(
        kind="call",
        iteration=3,
        attempt=7,
        started_at=datetime(2026, 10, 22, 1, 2, 3, tzinfo=UTC),
        finished_at=datetime(2026, 10, 22, 1, 2, 5, tzinfo=UTC),
        device_snapshot=None,
        outcome="completed",
        reason=None,
        stopped=False,
    )
    completed = workflows.Completed(source, False, facts)
    record = workflows.record_with_metadata(
        completed,
        identity=workflows.RunIdentity(
            "0123456789abcdef0123456789abcdef", history.provenance
        ),
        experiment_tag=core.AXES_SPEC.tag,
        snapshot=history.snapshot,
        completion=completion,
    )
    assert record.result is source.result
    content = RunMetadataContent(record.metadata)
    assert content.run_id == "0123456789abcdef0123456789abcdef-000003-7"
    assert content.experiment == core.AXES_SPEC.tag
    assert content.snapshot == history.snapshot
    assert content.started_at == "2026-10-22T01:02:03+00:00"
    assert content.finished_at == "2026-10-22T01:02:05+00:00"
    assert content.provenance == history.provenance
    assert source.metadata == {
        "device_snapshot": {"flux": {"value": 0.125}}, "author": "kept"
    }
    path = tmp_path / "measurement.h5"
    save_run(record, path, spec=core.AXES_SPEC)
    saved = datafile.load(path).docs["metadata"]
    assert isinstance(saved, dict)
    assert saved["completion"] == completion
    assert saved["device_snapshot"] == {"flux": {"value": 0.125}}
    assert saved["author"] == "kept"


def test_record_metadata_requires_explicit_completion() -> None:
    history = native_metadata()
    cfg = SavingCfg()
    source = RunRecord(cfg, SavingCore().run(cfg, context=None))
    facts = workflows.OperationFacts(
        kind="call", iteration=0, attempt=1,
        started_at=datetime(2026, 10, 22, tzinfo=UTC),
        finished_at=datetime(2026, 10, 22, tzinfo=UTC),
        device_snapshot=None, outcome="completed", reason=None, stopped=False,
    )
    assemble: Callable[..., object] = workflows.record_with_metadata
    with pytest.raises(TypeError, match="completion"):
        assemble(
            workflows.Completed(source, False, facts),
            identity=workflows.RunIdentity("run", history.provenance),
            experiment_tag="synthetic/trace", snapshot=history.snapshot,
        )
