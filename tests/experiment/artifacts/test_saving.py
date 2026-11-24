"""Shared saving through typed artifacts, real writers and a real entry ledger."""

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.datafile import (
    DataVariable,
    load,
    load_grouped_labber_data,
    load_labber_data,
    read_doc,
)
from zcu_tools.experiment import (
    AnalysisDocument,
    AnalysisItem,
    ArtifactRef,
    AxesSpec,
    DataItem,
    FigureItem,
    RunMetadata,
    RunRecord,
    RunValueAxis,
    RunValueSpec,
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
)

from tests._native_support import native_metadata
from tests.experiment._grouped_layout_support import (
    assert_mixed_disk_values,
    mixed_record,
    mixed_spec,
)
from tests.experiment._optional_variables_support import (
    diagnostic_record,
    diagnostic_spec,
    invalid_diagnostic_export_spec,
)
from tests.experiment._saving_support import (
    SavingCfg,
    SavingCore,
    SavingResult,
    saving_figure,
)


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
                entry_name=entry.result_path.name,
            )
        }
    )
    layout = SaveLayout(
        result_path=entry.result_path,
        database_path=entry.database_path,
        stem=f"Synthetic_Experiment_{metadata.run_id}",
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
    prepared = prepare_run_data(source, spec=core.AXES_SPEC, comment="raw")
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
        record=RunRecord(None, None, metadata=metadata.model_dump(mode="json")),
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
    with h5py.File(labber, "r") as file:
        comment = json.loads(str(file.attrs["comment"]))
    assert comment["snapshot"] == {
        "point": metadata.snapshot.point,
        "roles": dict(metadata.snapshot.roles),
    }
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
            record=RunRecord(None, None, metadata=metadata.model_dump(mode="json")),
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


def test_artifact_data_projects_labber_without_losing_native_diagnostic(
    tmp_path: Path,
) -> None:
    spec = replace(diagnostic_spec(), labber_variables=("data",))
    source = diagnostic_record(present=True)
    metadata, layout, ledger = storage(tmp_path)
    metadata = metadata.model_copy(update={"experiment": spec.tag})
    report = save_artifacts(
        (
            SaveArtifact(
                ArtifactRef("run", "measurement_data"),
                (
                    DataItem.from_run(
                        ArtifactKey("run", "measurement_data", "data"),
                        source,
                        spec=spec,
                        comment="diagnostic projection",
                    ),
                ),
            ),
        ),
        layout=layout,
        record=replace(source, metadata=metadata.model_dump(mode="json")),
        ledger=ledger,
        origin="notebook",
    )
    (native,) = (item for item in report.outputs if item.output.format == "data_h5")
    (labber,) = (item for item in report.outputs if item.output.format == "labber")
    assert native.status == "saved", native.error
    assert labber.status == "saved", labber.error
    stored = load(native.output.path)
    assert set(stored.vars) == {"data", "diagnostics"}
    np.testing.assert_array_equal(stored.vars["diagnostics"].values, [[2, 10], [3, 16]])
    exported = load_labber_data(str(labber.output.path))
    assert exported.data.name == "data"
    np.testing.assert_array_equal(exported.z, [1 + 2j, 3 - 4j])
    np.testing.assert_allclose(exported.axes[0].values, [1e-6, 2e-6])


def test_artifact_save_isolates_invalid_unselected_variable(tmp_path: Path) -> None:
    spec = invalid_diagnostic_export_spec()
    metadata, layout, ledger = storage(tmp_path)
    metadata = metadata.model_copy(update={"experiment": spec.tag})
    source = replace(
        diagnostic_record(present=True), metadata=metadata.model_dump(mode="json")
    )
    report = save_artifacts(
        (
            SaveArtifact(
                ArtifactRef("run", "measurement_data"),
                (
                    DataItem.from_run(
                        ArtifactKey("run", "measurement_data", "data"),
                        source,
                        spec=spec,
                        comment="invalid excluded diagnostic",
                    ),
                    FigureItem(
                        ArtifactKey("run", "measurement_data", "figure", "fit"),
                        saving_figure(),
                    ),
                ),
            ),
        ),
        layout=layout,
        record=source,
        ledger=ledger,
        origin="notebook",
    )

    failed = [item for item in report.outputs if item.status == "failed"]
    assert {item.output.format for item in failed} == {"data_h5", "labber"}
    for item in failed:
        assert item.error is not None
        assert f"{spec.tag}: /vars/diagnostics: duplicate axis" in item.error
        assert not item.output.path.exists()
    figures = [item for item in report.outputs if item.output.format == "png"]
    assert figures
    for item in figures:
        assert item.status == "saved", item.error
        assert item.output.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("analysis_only", [False, True])
def test_nonfinite_author_metadata_rejects_before_any_output_or_ledger(
    tmp_path: Path, value: float, analysis_only: bool
) -> None:
    metadata, layout, ledger = storage(tmp_path)
    document = metadata.model_dump(mode="json")
    document["author_notes"] = {"nested": [{"value": value}]}
    selected = artifacts()[1:] if analysis_only else artifacts()
    before = set(tmp_path.rglob("*"))

    with pytest.raises(ValueError, match="metadata"):
        save_artifacts(
            selected,
            layout=layout,
            record=RunRecord(None, None, metadata=document),
            ledger=ledger,
            origin="gui",
        )

    assert set(tmp_path.rglob("*")) == before
    assert ledger.events() == ()


@pytest.mark.parametrize("start", [0.25, float("inf")], ids=["valid", "nonfinite"])
def test_prepared_run_value_snapshot_and_failures_are_native_only(
    tmp_path: Path, start: float
) -> None:
    @dataclass
    class ResultWithFixedValues(SavingResult):
        start: float
        offsets: NDArray[np.float64]

    metadata, layout, ledger = storage(tmp_path)
    original = SavingCore.AXES_SPEC
    spec = AxesSpec(
        axes=original.axes,
        z=original.z,
        result_type=ResultWithFixedValues,
        cfg_type=original.cfg_type,
        tag=original.tag,
        run_values=(
            RunValueSpec(field_name="start", wire_name="start", unit="s"),
            RunValueSpec(
                field_name="offsets",
                wire_name="offsets",
                unit="Hz",
                dtype=np.float64,
                axis=RunValueAxis(variable=DataVariable("data"), label="frequency"),
            ),
        ),
    )
    cfg = SavingCfg()
    signal = SavingCore().run(cfg, context=None)
    record = RunRecord(
        cfg,
        ResultWithFixedValues(
            signal.frequency, signal.signal, start, np.array([1.0, 2.0, 3.0])
        ),
        metadata=metadata.model_dump(mode="json"),
    )
    item = DataItem.from_run(
        ArtifactKey("run", "measurement_data", "data"), record, spec=spec, comment=""
    )
    record.result.start = 0.5
    record.result.offsets[:] = [4.0, 5.0, 6.0]
    report = save_artifacts(
        (SaveArtifact(ArtifactRef("run", "measurement_data"), (item,)),),
        layout=layout,
        record=record,
        ledger=ledger,
        origin="gui",
    )
    outputs = {result.output.format: result for result in report.outputs}
    labber = outputs["labber"]
    assert labber.status == "saved"
    assert labber.output.path.is_file()
    native = outputs["data_h5"]
    if np.isfinite(start):
        assert native.status == "saved"
        assert read_doc(native.output.path, "run_values") == {
            "version": 1,
            "values": {
                "start": {"unit": "s", "value": 0.25},
                "offsets": {"unit": "Hz", "value": [1.0, 2.0, 3.0]},
            },
        }
    else:
        assert native.status == "failed"
        assert native.error and "/docs/run_values/values/start/value" in native.error
        assert not native.output.path.exists()


def test_one_failed_destination_does_not_block_other_outputs(tmp_path: Path) -> None:
    metadata, layout, ledger = storage(tmp_path)
    selected = artifacts()
    figure_output = layout.outputs(selected[1].members[0].key)[0]
    figure_output.path.parent.parent.mkdir(parents=True)
    figure_output.path.parent.write_text("block this PNG directory")
    report = save_artifacts(
        selected,
        layout=layout,
        record=RunRecord(None, None, metadata=metadata.model_dump(mode="json")),
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
        record=RunRecord(None, None, metadata=metadata.model_dump(mode="json")),
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
            record=RunRecord(None, None, metadata=metadata.model_dump(mode="json")),
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
            record=RunRecord(None, None, metadata=metadata.model_dump(mode="json")),
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
