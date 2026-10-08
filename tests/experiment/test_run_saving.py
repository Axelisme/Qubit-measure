"""Shared saving through typed artifacts, real writers and a real entry ledger."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from unittest.mock import Mock
from uuid import UUID

import pytest
from zcu_tools.datafile import (
    AnalysisDocument,
    RunMetadata,
    load_run_data,
)
from zcu_tools.experiment import (
    ArtifactRef,
    DataMember,
    FileMember,
    RunRecord,
    SaveArtifact,
    prepare_run_data,
    save_artifacts,
)
from zcu_tools.resources.entry import (
    ArtifactKey,
    RecordsLedger,
    ResultEntry,
    SavedPayload,
    SaveLayout,
    snapshot,
)

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore, saving_figure


def storage(tmp_path: Path) -> tuple[RunMetadata, SaveLayout, RecordsLedger]:
    entry = ResultEntry.create(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    entry.new_point("sweet")
    original = native_metadata("test/saving")
    metadata = replace(
        original,
        snapshot=replace(
            original.snapshot,
            entry_id=UUID(snapshot(entry.setup).general.entry_id),
            entry_name="entry",
        ),
    )
    layout = SaveLayout(
        result_path=entry.result_path,
        database_path=entry.database_path,
        run_id=metadata.run_id,
        point="sweet",
        saved_at=datetime.now(UTC),
    )
    return metadata, layout, entry.ledger


def artifacts() -> tuple[SaveArtifact, ...]:
    cfg = SavingCfg(gain=2.0)
    core = SavingCore()
    source = RunRecord(
        cfg=cfg,
        result=core.run(cfg, context=None),
    )
    prepared = prepare_run_data(source, spec=core.persistence_spec(), comment="raw")
    document = AnalysisDocument(
        pane="analysis", parameters={"skip": 1}, summary={"value": 3.0}
    )
    return (
        SaveArtifact(
            ArtifactRef("run", "measurement_data"),
            (DataMember(ArtifactKey("run", "measurement_data", "data"), prepared),),
        ),
        SaveArtifact(
            ArtifactRef("analysis", "results"),
            (
                FileMember(
                    ArtifactKey("analysis", "results", "figure", "fit"), saving_figure()
                ),
                FileMember(
                    ArtifactKey("analysis", "results", "analysis", "analysis"),
                    document,
                ),
            ),
        ),
    )


@pytest.mark.parametrize("origin", ["gui", "notebook", "mcp"])
def test_all_outputs_share_content_context_and_one_saved_event(
    tmp_path: Path,
    origin: Literal["gui", "notebook", "mcp"],
) -> None:
    metadata, layout, ledger = storage(tmp_path)
    report = save_artifacts(
        artifacts(),
        layout=layout,
        metadata=metadata,
        ledger=ledger,
        origin=origin,
    )
    assert ledger.events()[-1].origin == origin
    assert report.run_id == metadata.run_id
    assert len(report.outputs) == 5
    assert all(item.status == "saved" and item.error is None for item in report.outputs)
    png_paths = [
        item.output.path for item in report.outputs if item.output.format == "png"
    ]
    assert png_paths[0].read_bytes() == png_paths[1].read_bytes()
    native = next(
        item.output.path for item in report.outputs if item.output.format == "data_h5"
    )
    labber = next(
        item.output.path for item in report.outputs if item.output.format == "labber"
    )
    loaded = load_run_data(native)
    assert loaded.metadata.snapshot == metadata.snapshot
    assert loaded.metadata.started_at == metadata.started_at
    assert loaded.metadata.labber_path == str(labber)
    assert loaded.cfg.values["gain"] == 2.0
    events = ledger.events()
    assert len(events) == 1
    assert isinstance(events[0].payload, SavedPayload)
    assert len(events[0].payload.outputs) == len(report.outputs)
    assert ledger.get(events[0].id).record is not None


def test_one_failed_destination_does_not_block_other_outputs(tmp_path: Path) -> None:
    metadata, layout, ledger = storage(tmp_path)
    selected = artifacts()
    figure_output = layout.outputs(selected[1].members[0].key)[0]
    figure_output.path.parent.parent.mkdir(parents=True)
    figure_output.path.parent.write_text("block this PNG directory")
    report = save_artifacts(
        selected,
        layout=layout,
        metadata=metadata,
        ledger=ledger,
        origin="notebook",
    )
    failed = [item for item in report.outputs if item.status == "failed"]
    assert len(failed) == 1
    assert failed[0].output == figure_output
    assert failed[0].error
    assert all(
        item.output.path.is_file() for item in report.outputs if item.status == "saved"
    )
    event = ledger.events()[0]
    assert isinstance(event.payload, SavedPayload)
    assert len([item for item in event.payload.outputs if item.status == "failed"]) == 1


def test_figure_is_encoded_once_for_equivalent_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, layout, ledger = storage(tmp_path)
    figure = saving_figure()
    render = Mock(wraps=figure.savefig)
    monkeypatch.setattr(figure, "savefig", render)
    selected = (
        artifacts()[0],
        SaveArtifact(
            ArtifactRef("analysis", "results"),
            (FileMember(ArtifactKey("analysis", "results", "figure", "fit"), figure),),
        ),
    )
    report = save_artifacts(
        selected,
        layout=layout,
        metadata=metadata,
        ledger=ledger,
        origin="notebook",
    )
    assert all(item.status == "saved" for item in report.outputs)
    pngs = [item.output.path for item in report.outputs if item.output.format == "png"]
    assert len(pngs) == 2
    assert pngs[0].read_bytes() == pngs[1].read_bytes()
    assert render.call_count == 1


@pytest.mark.parametrize("failing_member", ["figure", "analysis"])
def test_encoding_failure_isolates_the_member_and_records_other_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_member: str
) -> None:
    metadata, layout, ledger = storage(tmp_path)
    figure = saving_figure()
    document = AnalysisDocument(pane="analysis", parameters={}, summary={"value": 1.0})
    if failing_member == "figure":
        monkeypatch.setattr(
            figure, "savefig", Mock(side_effect=ValueError("PNG encoder failed"))
        )
    else:
        document.summary["value"] = float("nan")
    selected = (
        artifacts()[0],
        SaveArtifact(
            ArtifactRef("analysis", "results"),
            (
                FileMember(ArtifactKey("analysis", "results", "figure", "fit"), figure),
                FileMember(
                    ArtifactKey("analysis", "results", "analysis", "analysis"), document
                ),
            ),
        ),
    )
    report = save_artifacts(
        selected,
        layout=layout,
        metadata=metadata,
        ledger=ledger,
        origin="notebook",
    )
    failed = [item for item in report.outputs if item.status == "failed"]
    assert len(failed) == (2 if failing_member == "figure" else 1)
    assert all(item.key.member == failing_member and item.error for item in failed)
    successful = [item for item in report.outputs if item.status == "saved"]
    assert all(item.output.path.is_file() for item in successful)
    assert {"data_h5", "labber"} <= {item.output.format for item in successful}
    event = ledger.events()[0]
    assert isinstance(event.payload, SavedPayload)
    assert len(
        [item for item in event.payload.outputs if item.status == "failed"]
    ) == len(failed)


def test_member_contract_rejects_invalid_identity_before_output_attempts() -> None:
    with pytest.raises(ValueError, match="section|path|segment"):
        ArtifactRef("..", "bad")
    with pytest.raises(ValueError, match="figure|analysis|member"):
        FileMember(ArtifactKey("run", "measurement_data", "data"), saving_figure())
    document = AnalysisDocument(pane="analysis", parameters={}, summary={})
    with pytest.raises(TypeError, match="Figure|figure|content"):
        FileMember(ArtifactKey("analysis", "results", "figure", "fit"), document)
    with pytest.raises(TypeError, match="AnalysisDocument|analysis|content"):
        FileMember(
            ArtifactKey("analysis", "results", "analysis", "analysis"), saving_figure()
        )
    member = FileMember(
        ArtifactKey("analysis", "results", "figure", "fit"), saving_figure()
    )
    with pytest.raises(ValueError, match="matching|section|artifact|ref"):
        SaveArtifact(ArtifactRef("post_analysis", "results"), (member,))
    with pytest.raises(ValueError, match="duplicate|unique"):
        SaveArtifact(ArtifactRef("analysis", "results"), (member, member))
