"""Shared saving through typed artifacts, real writers and a real entry ledger."""

from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.datafile import (
    load,
    load_grouped_labber_data,
)
from zcu_tools.experiment import (
    AnalysisDocument,
    AnalysisItem,
    ArtifactRef,
    DataItem,
    FigureItem,
    RunMetadata,
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
from tests.experiment._grouped_layout_support import (
    assert_mixed_disk_values,
    mixed_record,
    mixed_spec,
)
from tests.experiment._saving_support import SavingCfg, SavingCore, saving_figure


def storage(tmp_path: Path) -> tuple[RunMetadata, SaveLayout, RecordsLedger]:
    entry = ResultEntry.create(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    entry.new_point("sweet")
    original = native_metadata("test/saving")
    metadata = original.model_copy(
        update={
            "snapshot": replace(
                original.snapshot,
                entry_id=UUID(snapshot(entry.setup).general.entry_id),
                entry_name="entry",
            )
        }
    )
    layout = SaveLayout(
        result_path=entry.result_path,
        database_path=entry.database_path,
        run_id=metadata.run_id,
        experiment="Synthetic/Experiment",
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
            (DataItem(ArtifactKey("run", "measurement_data", "data"), prepared),),
        ),
        SaveArtifact(
            ArtifactRef("analysis", "results"),
            (
                FigureItem(
                    ArtifactKey("analysis", "results", "figure", "fit"), saving_figure()
                ),
                AnalysisItem(
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
        record=RunRecord(None, None, metadata=metadata),
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
    run = tmp_path / "Database/entry/runs" / f"Synthetic_Experiment_{metadata.run_id}"
    assert native == run / "data.h5"
    assert png_paths == [
        run / "figures/analysis_results_fit.png",
        tmp_path
        / "results/entry/points/sweet/figures"
        / f"Synthetic_Experiment_{metadata.run_id}_analysis_results_fit.png",
    ]
    analysis = next(
        item.output.path for item in report.outputs if item.output.format == "json"
    )
    assert analysis == run / "analysis/analysis.json"
    assert analysis.is_file()
    loaded = load(native)
    assert loaded.docs["metadata"] == metadata.model_dump(mode="json")
    assert loaded.docs["cfg"]["values"] == {"dev": None, "gain": 2.0}
    events = ledger.events()
    assert len(events) == 1
    assert isinstance(events[0].payload, SavedPayload)
    assert events[0].payload.labber_path == str(labber)
    assert len(events[0].payload.outputs) == len(report.outputs)
    assert ledger.get(events[0].id).record is not None


@pytest.mark.parametrize(
    "independent", [True, False], ids=["independent", "default-common"]
)
def test_grouped_artifacts_obey_declared_layout_and_preserve_native_ledger_semantics(
    tmp_path: Path, independent: bool
) -> None:
    spec = mixed_spec(independent=independent)
    source = mixed_record()
    metadata, layout, ledger = storage(tmp_path)
    metadata = metadata.model_copy(update={"experiment": spec.tag})
    selected = (
        SaveArtifact(
            ArtifactRef("run", "measurement_data"),
            (
                DataItem.from_run(
                    ArtifactKey("run", "measurement_data", "data"),
                    source,
                    spec=spec,
                    comment="mixed rank export",
                ),
            ),
        ),
    )
    labber_paths: list[Path] = []
    native_paths: list[Path] = []
    for _ in range(2):
        report = save_artifacts(
            selected,
            layout=layout,
            record=RunRecord(None, None, metadata=metadata),
            ledger=ledger,
            origin="notebook",
        )
        (native,) = (item for item in report.outputs if item.output.format == "data_h5")
        (labber,) = (item for item in report.outputs if item.output.format == "labber")
        assert native.status == "saved" and native.error is None
        assert labber.status == ("saved" if independent else "failed")
        if independent:
            assert labber.error is None
            loaded = load_grouped_labber_data(
                str(labber.output.path), required_variables=spec.required_variables
            )
            assert_mixed_disk_values(loaded.variables)
        else:
            assert labber.error and "common grid" in labber.error
            assert not labber.output.path.exists()
        stored = load(native.output.path)
        assert set(stored.vars) == {"signals", "detunes"}
        np.testing.assert_allclose(
            stored.vars["signals"].values, source.result.signals * 1e-3
        )
        np.testing.assert_array_equal(stored.vars["signals"].axes[0].values, [2, 5])
        np.testing.assert_allclose(
            stored.vars["signals"].axes[1].values, [1e-6, 2e-6, 4e-6]
        )
        np.testing.assert_allclose(
            stored.vars["detunes"].values, [250_000.0, -500_000.0]
        )
        assert stored.docs["metadata"] == metadata.model_dump(mode="json")
        assert stored.docs["cfg"]["values"] == {"name": "mixed grids"}
        event = ledger.events()[-1]
        assert isinstance(event.payload, SavedPayload)
        assert event.payload.labber_path == (
            str(labber.output.path) if independent else None
        )
        assert [
            (item.format, item.status, item.path) for item in event.payload.outputs
        ] == [
            (item.output.format, item.status, str(item.output.path))
            for item in report.outputs
        ]
        labber_paths.append(labber.output.path)
        native_paths.append(native.output.path)
    assert native_paths[0] == native_paths[1]
    if independent:
        assert labber_paths[0] != labber_paths[1]
    assert len(ledger.events()) == 2


def test_one_failed_destination_does_not_block_other_outputs(tmp_path: Path) -> None:
    metadata, layout, ledger = storage(tmp_path)
    selected = artifacts()
    figure_output = layout.outputs(selected[1].members[0].key)[0]
    figure_output.path.parent.parent.mkdir(parents=True)
    figure_output.path.parent.write_text("block this PNG directory")
    report = save_artifacts(
        selected,
        layout=layout,
        record=RunRecord(None, None, metadata=metadata),
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


def test_nonfinite_analysis_isolates_the_member_and_records_other_outputs(
    tmp_path: Path,
) -> None:
    metadata, layout, ledger = storage(tmp_path)
    figure = saving_figure()
    document = AnalysisDocument(pane="analysis", parameters={}, summary={"value": 1.0})
    document.summary["value"] = float("nan")
    selected = (
        artifacts()[0],
        SaveArtifact(
            ArtifactRef("analysis", "results"),
            (
                FigureItem(ArtifactKey("analysis", "results", "figure", "fit"), figure),
                AnalysisItem(
                    ArtifactKey("analysis", "results", "analysis", "analysis"), document
                ),
            ),
        ),
    )
    report = save_artifacts(
        selected,
        layout=layout,
        record=RunRecord(None, None, metadata=metadata),
        ledger=ledger,
        origin="notebook",
    )
    failed = [item for item in report.outputs if item.status == "failed"]
    assert len(failed) == 1
    assert all(item.key.member == "analysis" and item.error for item in failed)
    successful = [item for item in report.outputs if item.status == "saved"]
    assert all(item.output.path.is_file() for item in successful)
    assert {"data_h5", "labber"} <= {item.output.format for item in successful}
    event = ledger.events()[0]
    assert isinstance(event.payload, SavedPayload)
    assert len(
        [item for item in event.payload.outputs if item.status == "failed"]
    ) == len(failed)


def test_injected_figure_encoder_runs_once_and_shares_png_bytes(tmp_path: Path) -> None:
    metadata, layout, ledger = storage(tmp_path)
    selected = artifacts()
    figure = next(
        member.content
        for artifact in selected
        for member in artifact.members
        if isinstance(member, FigureItem)
    )
    calls: list[Figure] = []
    encoded: list[bytes] = []

    def encode(candidate: Figure) -> bytes:
        calls.append(candidate)
        with BytesIO() as stream:
            candidate.savefig(stream, format="png", dpi=37)
            content = stream.getvalue()
        encoded.append(content)
        return content

    try:
        report = save_artifacts(
            selected,
            layout=layout,
            record=RunRecord(None, None, metadata=metadata),
            ledger=ledger,
            origin="gui",
            figure_encoder=encode,
        )
        assert calls == [figure]
        pngs = [
            item.output.path for item in report.outputs if item.output.format == "png"
        ]
        assert len(pngs) == 2
        assert all(path.read_bytes() == encoded[0] for path in pngs)
        assert all(item.status == "saved" for item in report.outputs)
    finally:
        figure.clear()


def test_injected_encoder_failure_leaves_data_and_json_saved(tmp_path: Path) -> None:
    metadata, layout, ledger = storage(tmp_path)
    selected = artifacts()
    figure = next(
        member.content
        for artifact in selected
        for member in artifact.members
        if isinstance(member, FigureItem)
    )
    calls: list[Figure] = []

    def encode(candidate: Figure) -> bytes:
        calls.append(candidate)
        raise RuntimeError("renderer unavailable")

    try:
        report = save_artifacts(
            selected,
            layout=layout,
            record=RunRecord(None, None, metadata=metadata),
            ledger=ledger,
            origin="gui",
            figure_encoder=encode,
        )
        assert calls == [figure]
        failed = [item for item in report.outputs if item.status == "failed"]
        assert len(failed) == 2
        assert all(item.output.format == "png" for item in failed)
        assert all(
            item.error and "renderer unavailable" in item.error for item in failed
        )
        successful = [item for item in report.outputs if item.status == "saved"]
        assert {"data_h5", "labber", "json"} <= {
            item.output.format for item in successful
        }
        assert all(item.output.path.is_file() for item in successful)
        event = ledger.events()[0]
        assert isinstance(event.payload, SavedPayload)
        assert (
            len([item for item in event.payload.outputs if item.status == "failed"])
            == 2
        )
    finally:
        figure.clear()


def test_member_contract_rejects_invalid_identity_before_output_attempts() -> None:
    with pytest.raises(ValueError, match="section|path|segment"):
        ArtifactRef("..", "bad")
    with pytest.raises(ValueError, match="figure|analysis|member"):
        FigureItem(ArtifactKey("run", "measurement_data", "data"), saving_figure())
    member = FigureItem(
        ArtifactKey("analysis", "results", "figure", "fit"), saving_figure()
    )
    document = AnalysisDocument(pane="analysis", parameters={}, summary={})
    analysis = AnalysisItem(
        ArtifactKey("analysis", "results", "analysis", "analysis"), document
    )
    with pytest.raises(TypeError, match="Figure|figure|content"):
        replace(member, content=document)
    with pytest.raises(TypeError, match="AnalysisDocument|analysis|content"):
        replace(analysis, content=saving_figure())
    with pytest.raises(ValueError, match="matching|section|artifact|ref"):
        SaveArtifact(ArtifactRef("post_analysis", "results"), (member,))
    with pytest.raises(ValueError, match="duplicate|unique"):
        SaveArtifact(ArtifactRef("analysis", "results"), (member, member))
