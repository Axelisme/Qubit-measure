"""Shared saved-input diagnostics using synthetic envelopes, not experiments."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import EncodedRecord, RunIdentity, RunPaths
from zcu_tools.experiment.workflows.journal import ErrorInfo
from zcu_tools.experiment.workflows.saved_run import (
    SavedCommit,
    SavedDiscarded,
    SavedFailed,
    SavedFinished,
    SavedIteration,
    SavedManifest,
    SavedStarted,
    read_points,
)


class SyntheticPoint(BaseModel):
    """Minimal committed record for the shared reader's typed output."""

    index: int


@pytest.fixture
def manifest(tmp_path: Path) -> SavedManifest:
    """Supply two missing summaries beside a third, writable invocation."""
    roots = RunPaths(tmp_path / "metadata", tmp_path / "data")
    for call_seq in (1, 2, 3):
        (roots.data_root / "iter" / f"{call_seq:06d}").mkdir(parents=True)
    now = datetime(2026, 10, 12, tzinfo=timezone.utc)
    return SavedManifest(
        format="zcu-workflow-run",
        format_version=1,
        identity=RunIdentity(
            "synthetic-run",
            SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname=None,
            ),
        ),
        workflow="synthetic",
        plan={"points": 1},
        initial_tunables={},
        roots=roots,
        journal=roots.metadata_root / "journal.jsonl",
        started_at=now,
        updated_at=now,
        ended_at=now,
        lifecycle="done",
        reason=None,
    )


def write_summary(
    manifest: SavedManifest,
    result: SavedCommit | SavedFinished | SavedDiscarded | SavedFailed,
) -> Path:
    """Serialize a valid third invocation through the public saved envelope."""
    source = manifest.roots.data_root / "iter" / "000003" / "record.json"
    saved = SavedIteration(
        format="zcu-workflow-iteration",
        format_version=1,
        identity=manifest.identity,
        workflow=manifest.workflow,
        plan=manifest.plan,
        tunables=manifest.initial_tunables,
        started=SavedStarted(
            kind="iteration_started", call_seq=3, commit_seq=0, revision=0
        ),
        result=result,
    )
    source.write_text(saved.model_dump_json(), encoding="utf-8")
    return source


@pytest.mark.parametrize(
    "result",
    [
        SavedFinished(
            kind="step_finished",
            call_seq=3,
            revision=0,
            iteration_dir="iter/000003",
            outcome="done",
            reason=None,
        ),
        SavedDiscarded(
            kind="step_discarded",
            call_seq=3,
            revision=0,
            iteration_dir="iter/000003",
            reason="pause",
        ),
        SavedFailed(
            kind="step_failed",
            call_seq=3,
            revision=0,
            iteration_dir="iter/000003",
            error=ErrorInfo("RuntimeError", "synthetic failure", "synthetic trace"),
        ),
    ],
    ids=["finished", "discarded", "failed"],
)
def test_missing_summaries_are_named_when_only_noncommit_records_remain(
    manifest: SavedManifest, result: SavedFinished | SavedDiscarded | SavedFailed
) -> None:
    write_summary(manifest, result)

    with pytest.raises(ValueError, match=r"missing record\.json") as raised:
        read_points(manifest, TypeAdapter(SyntheticPoint))

    message = str(raised.value)
    for call_seq in (1, 2):
        missing = manifest.roots.data_root / "iter" / f"{call_seq:06d}" / "record.json"
        assert str(missing) in message


def test_committed_point_and_all_missing_summaries_are_returned_together(
    manifest: SavedManifest,
) -> None:
    point = SyntheticPoint(index=0)
    source = write_summary(
        manifest,
        SavedCommit(
            kind="step_committed",
            call_seq=3,
            revision=0,
            iteration_dir="iter/000003",
            commit_seq=1,
            encoded_record=EncodedRecord("json", point.model_dump(mode="json"), None),
        ),
    )

    points, missing = read_points(manifest, TypeAdapter(SyntheticPoint))

    assert tuple(saved.point for saved in points) == (point,)
    assert tuple(saved.record_path for saved in points) == (source,)
    assert missing == (
        manifest.roots.data_root / "iter" / "000001" / "record.json",
        manifest.roots.data_root / "iter" / "000002" / "record.json",
    )
