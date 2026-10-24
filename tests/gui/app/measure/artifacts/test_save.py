"""Synchronous native Save through real state, writers and ledger."""

import json
import threading
from collections.abc import Generator
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import Container, load
from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.artifacts import (
    ArtifactSectionSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.measure.controller import Controller, DiagnosticSink
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    PointUseRequest,
)
from zcu_tools.resources.entry import (
    ResultEntry,
    SavedPayload,
    SaveLayout,
)

from tests.experiment._saving_support import SavingCfg, SavingCore
from tests.gui.app.measure._native_save_support import (
    GUI_ORIGIN,
    NativeSaveSession,
    PaneParams,
    PaneSummary,
    PostSummary,
    declare_panes,
)
from tests.gui.app.measure._record_support import require_record
from tests.gui.app.measure.services._native_acquisition_support import RoleSavingAdapter


@pytest.mark.parametrize("failure", ["missing_cfg", "invalid_shape"])
def test_data_preparation_failure_reports_outputs_and_saves_other_artifacts(
    native_session: NativeSaveSession, failure: str
) -> None:
    session = native_session
    cfg = SavingCfg()
    result = SavingCore().run(cfg, context=None)
    if failure == "invalid_shape":
        result.signal = result.signal[:1]
    source = RunRecord(cfg=None if failure == "missing_cfg" else cfg, result=result)
    session.state.update_tab_result(
        "tab",
        replace(source, metadata=session.metadata),
        parameters=session.state.get_tab("tab").run.parameters,
    )
    declare_panes(session)
    try:
        report = session.save_artifacts()
    except (TypeError, ValueError) as error:
        pytest.fail(f"Data preparation aborted independent artifacts: {error}")
    data = [item for item in report.outputs if item.key.member == "data"]
    other = [item for item in report.outputs if item.key.member != "data"]
    assert len(data) == 2
    assert all(item.status == "failed" and item.error for item in data)
    assert all(not item.output.path.exists() for item in data)
    assert len(other) == 6
    assert all(item.status == "saved" and item.output.path.is_file() for item in other)
    assert session.artifact_completions[-1].report == report
    assert session.entry.ledger.events()[-1].kind == "saved"


def test_data_save_writes_both_formats_and_resave_keeps_run_identity(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    assert session.state.get_tab("tab").run.confirmed_run_id is None
    planned = session.state.get_artifact_snapshots("tab")
    assert (
        planned[0].artifacts[0].members[0].outputs[1].path.name
        == "saving_a1b2c3_1.hdf5"
    )
    first = session.save()
    assert session.state.get_tab("tab").run.confirmed_run_id == first.run_id
    native, labber = first.outputs
    assert native.status == labber.status == "saved"
    assert labber.output.path.name == "saving_a1b2c3_1.hdf5"
    assert (
        native.output.path
        == session.entry.database_path
        / "runs"
        / f"saving_{session.metadata.run_id}"
        / "data.h5"
    )
    stored = load(native.output.path)
    assert metadata_from_container(stored).run_id == session.metadata.run_id
    assert metadata_from_container(stored).started_at == session.metadata.started_at
    assert metadata_from_container(stored).snapshot == session.metadata.snapshot
    variable = next(iter(stored.vars.values()))
    np.testing.assert_allclose(variable.axes[0].values, [1e9, 1.001e9, 1.002e9])
    second = session.save(origin=CommandOrigin("mcp"))
    assert second.run_id == first.run_id
    assert second.outputs[0].output.path == native.output.path
    assert second.outputs[1].output.path.name == "saving_a1b2c3_2.hdf5"
    events = session.entry.ledger.events()
    assert len(events) == 2
    assert [event.origin for event in events] == ["gui", "mcp"]
    assert isinstance(events[-1].payload, SavedPayload)
    assert [output.status for output in events[-1].payload.outputs] == [
        "saved",
        "saved",
    ]


@pytest.mark.parametrize("blocked", ["native", "labber"])
def test_each_output_failure_preserves_the_other_and_reports_both(
    native_session: NativeSaveSession, blocked: str
) -> None:
    session = native_session
    if blocked == "native":
        path = (
            session.entry.database_path
            / "runs"
            / f"saving_{session.metadata.run_id}"
            / "data.h5"
        )
        path.mkdir(parents=True)
    else:
        session.entry.database_path.mkdir(parents=True, exist_ok=True)
        (session.entry.database_path / "Labber").write_text("not a directory")
    completed = session.save()
    native, labber = completed.outputs
    assert [native.status, labber.status] == (
        ["failed", "saved"] if blocked == "native" else ["saved", "failed"]
    )
    failed = native if blocked == "native" else labber
    saved = labber if blocked == "native" else native
    assert saved.output.path.is_file()
    assert session.state.get_tab("tab").run.confirmed_run_id == completed.run_id
    assert failed.error
    event = session.entry.ledger.events()[-1]
    assert isinstance(event.payload, SavedPayload)
    assert [output.status for output in event.payload.outputs] == [
        native.status,
        labber.status,
    ]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    ("comment", "expected"),
    [(None, "retained note"), ("", ""), ("new note", "new note")],
)
def test_save_commands_preserve_clear_or_replace_annotation_in_native_output(
    native_session: NativeSaveSession,
    batch: bool,
    comment: str | None,
    expected: str,
) -> None:
    session = native_session
    session.control.set_comment("tab", "retained note")
    command = session.control.save_artifacts if batch else session.control.save_data
    report = command("tab", comment=comment, origin=GUI_ORIGIN)

    assert session.state.get_tab("tab").save.comment == expected
    native = next(item for item in report.outputs if item.output.format == "data_h5")
    assert native.status == "saved"


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("output_failure", [False, True])
def test_save_events_observe_committed_state_and_complete_report(
    native_session: NativeSaveSession, batch: bool, output_failure: bool
) -> None:
    session = native_session
    declare_panes(session)
    if output_failure:
        session.entry.database_path.mkdir(parents=True, exist_ok=True)
        (session.entry.database_path / "Labber").write_text("not a directory")
    completions: list[
        tuple[
            SaveDataFinishedPayload | SaveArtifactsFinishedPayload,
            tuple[ArtifactSectionSnapshot, ...],
            str,
        ]
    ] = []
    facts: list[
        tuple[TabInteractionChangedPayload, tuple[ArtifactSectionSnapshot, ...]]
    ] = []

    def completed(
        payload: SaveDataFinishedPayload | SaveArtifactsFinishedPayload,
    ) -> None:
        completions.append(
            (
                payload,
                session.state.get_artifact_snapshots(payload.tab_id),
                session.state.get_tab(payload.tab_id).save.comment,
            )
        )

    def interaction(payload: TabInteractionChangedPayload) -> None:
        facts.append((payload, session.state.get_artifact_snapshots(payload.tab_id)))

    completion_subscription = session.bus.subscribe(
        SaveArtifactsFinishedPayload if batch else SaveDataFinishedPayload, completed
    )
    fact_subscription = session.bus.subscribe(TabInteractionChangedPayload, interaction)
    try:
        command = session.control.save_artifacts if batch else session.control.save_data
        report = command("tab", comment="committed note", origin=GUI_ORIGIN)
    finally:
        completion_subscription.unsubscribe()
        fact_subscription.unsubscribe()

    assert len(completions) == 1
    payload, at_completion, annotation = completions[0]
    committed = session.state.get_artifact_snapshots("tab")
    assert payload.tab_id == "tab"
    assert payload.report == report
    assert at_completion == committed
    assert annotation == "committed note"
    assert len(report.outputs) == (8 if batch else 2)
    expected_fact = (
        TabInteractionFact.SAVE_FAILED
        if output_failure
        else TabInteractionFact.SAVE_SUCCEEDED
    )
    save_facts = [item for item in facts if item[0].fact is expected_fact]
    assert len(save_facts) == 1
    assert save_facts[0][0].tab_id == "tab"
    assert save_facts[0][1] == committed
    committed_outputs = {
        (member.key, output.path): output.status
        for section in committed
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    }
    assert all(
        committed_outputs[(item.key, item.output.path)] == item.status
        for item in report.outputs
    )
    if output_failure:
        failed = [item for item in report.outputs if item.status == "failed"]
        assert len(failed) == 1
        assert payload.error is not None
        assert str(failed[0].output.path) in payload.error
        assert failed[0].error and failed[0].error in payload.error
    else:
        assert payload.error is None
        assert all(item.status == "saved" for item in report.outputs)
    if isinstance(payload, SaveDataFinishedPayload):
        native = next(
            item for item in report.outputs if item.output.format == "data_h5"
        )
        assert payload.data_path == str(native.output.path)
    assert len(session.artifact_completions if batch else session.completions) == 1
    assert (session.completions if batch else session.artifact_completions) == []


@pytest.fixture
def role_controller(
    tmp_path: Path, qapp: QApplication
) -> Generator[tuple[State, Controller, BaseEventBus]]:
    """Compose the real diagnostic owner without hardware or mutable registries."""
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    registry = Registry()
    registry.register("role_saving", RoleSavingAdapter)
    controller = Controller(
        state=state, registry=registry, view=None, bus=bus, project_root=str(tmp_path)
    )
    try:
        yield state, controller, bus
    finally:
        controller.close()
        qapp.processEvents()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("output_failure", [False, True])
def test_controller_projects_native_save_completion_to_diagnostic_sinks(
    native_session: NativeSaveSession,
    role_controller: tuple[State, Controller, BaseEventBus],
    batch: bool,
    output_failure: bool,
) -> None:
    session = native_session
    declare_panes(session)
    if output_failure:
        session.entry.database_path.mkdir(parents=True, exist_ok=True)
        (session.entry.database_path / "Labber").write_text("not a directory")
    report = session.save_artifacts() if batch else session.save()
    payload = (session.artifact_completions if batch else session.completions)[-1]
    assert payload.report == report

    _, controller, bus = role_controller
    sink = MagicMock(spec=DiagnosticSink)
    controller.add_diagnostic_sink(sink)
    try:
        bus.emit(payload)
    finally:
        controller.remove_diagnostic_sink(sink)

    diagnostics = [call.args for call in sink.notify_diagnostic.call_args_list]
    # Every completion gives one summary line; failures add the full detail.
    expected_severities = ["info", "error"] if output_failure else ["info"]
    assert [item[0] for item in diagnostics] == expected_severities
    summary = diagnostics[0][2]
    assert report.run_id in summary
    assert f"{len(report.outputs)} files" in summary
    if output_failure:
        assert payload.error is not None
        errors = [item[2] for item in diagnostics if item[0] == "error"]
        for item in report.outputs:
            if item.error is not None:
                assert item.error in errors[0]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("rejection", ["busy", "no_run_result"])
def test_save_commands_reject_before_annotation_or_output_mutation(
    native_session: NativeSaveSession, batch: bool, rejection: str
) -> None:
    session = native_session
    session.control.set_comment("tab", "retained note")
    if rejection == "busy":
        session.state.set_tab_analyzing("tab", analyzing=True)
    else:
        session.state.clear_tab_results("tab")
    before = session.state.get_artifact_snapshots("tab")
    command = session.control.save_artifacts if batch else session.control.save_data
    with pytest.raises(FailedPreconditionError, match="busy|No run result"):
        command("tab", comment="rejected note", origin=GUI_ORIGIN)
    assert session.state.get_tab("tab").save.comment == "retained note"
    assert session.state.get_artifact_snapshots("tab") == before
    assert session.state.is_tab_busy("tab") is (rejection == "busy")
    assert not session.completions and not session.artifact_completions
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()
    assert not (session.entry.database_path / "Labber").exists()


def test_saved_analysis_selection_resaves_without_writing_other_sections(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session)
    first = session.save_artifacts()
    before = session.state.get_artifact_snapshots("tab")
    assert before[1].artifacts[0].status is SaveStatus.SAVED
    unselected = {
        item.output.path: item.output.path.stat().st_mtime_ns
        for item in first.outputs
        if item.key.section != "analysis"
    }
    report = session.save_artifacts((ArtifactRef("analysis", "results"),))
    assert len(report.outputs) == 5
    assert all(
        item.key.section == "analysis" and item.status == "saved"
        for item in report.outputs
    )
    after = session.state.get_artifact_snapshots("tab")
    assert after[0] == before[0] and after[2] == before[2]
    assert after[1].artifacts[0].status is SaveStatus.SAVED
    assert {path: path.stat().st_mtime_ns for path in unselected} == unselected
    saved = session.entry.ledger.events()[-1].payload
    assert isinstance(saved, SavedPayload)
    assert len(saved.outputs) == 5
    assert [(item.path, item.status) for item in saved.outputs] == [
        (str(item.output.path), item.status) for item in report.outputs
    ]


def test_first_analysis_save_all_failed_preserves_run_and_post_sections(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session)
    sections = session.state.get_artifact_snapshots("tab")
    assert all(
        artifact.status is SaveStatus.NOT_SAVED
        for section in sections
        for artifact in section.artifacts
    )
    for member in sections[1].artifacts[0].members:
        for output in member.outputs:
            output.path.mkdir(parents=True)
    report = session.save_artifacts()
    analysis = [item for item in report.outputs if item.key.section == "analysis"]
    assert len(analysis) == 5
    assert all(item.status == "failed" and item.error for item in analysis)
    assert all(
        item.status == "saved" and item.output.path.is_file()
        for item in report.outputs
        if item.key.section != "analysis"
    )
    committed = session.state.get_artifact_snapshots("tab")
    assert [section.section for section in committed] == [
        "run",
        "analysis",
        "post_analysis",
    ]
    assert committed[1].artifacts[0].status is SaveStatus.NOT_SAVED
    assert all(
        output.status == "failed" and output.error
        for member in committed[1].artifacts[0].members
        for output in member.outputs
    )
    assert committed[0].artifacts[0].status is SaveStatus.SAVED
    assert committed[2].artifacts[0].status is SaveStatus.SAVED
    assert session.state.get_tab("tab").run.confirmed_run_id == report.run_id
    assert session.artifact_completions[-1].report == report
    assert session.artifact_completions[-1].error


def test_save_control_has_tab_observes_existing_and_missing_resources(
    native_session: NativeSaveSession,
) -> None:
    assert native_session.control.has_tab("tab") is True
    assert native_session.control.has_tab("missing") is False


def test_save_uses_tab_storage_after_live_entry_switch(
    native_session: NativeSaveSession, tmp_path: Path
) -> None:
    session = native_session
    other = ResultEntry.create(
        "other", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    other.new_point("q")
    session.context.install_entry(other)
    ContextControlFacet(context=session.context).use_point(PointUseRequest("q"))
    completed = session.save()
    assert all(
        item.output.path.is_relative_to(session.entry.database_path)
        for item in completed.outputs
    )
    assert len(session.entry.ledger.events()) == 1
    assert other.ledger.events() == ()


@pytest.mark.parametrize("missing", ["metadata", "storage"])
def test_missing_native_evidence_rejects_before_operation_or_files(
    native_session: NativeSaveSession, missing: str
) -> None:
    session = native_session
    if missing == "metadata":
        session.state.update_tab_result(
            "tab",
            replace(require_record(session.state.get_tab("tab").run), metadata=None),
        )
    else:
        session.state.get_tab("tab").run.storage = None
    with pytest.raises(FailedPreconditionError):
        session.control.save_data("tab", origin=GUI_ORIGIN)
    assert not session.state.is_tab_busy("tab")
    assert session.completions == []
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


def test_invalid_record_reports_failed_data_outputs_and_saved_event(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    session.state.update_tab_result(
        "tab",
        RunRecord(cfg=None, result=object(), metadata=session.metadata),
        parameters=session.state.get_tab("tab").run.parameters,
    )
    report = session.save()
    assert len(report.outputs) == 2
    assert all(item.status == "failed" for item in report.outputs)
    assert all(item.error and "RunRecord" in item.error for item in report.outputs)
    assert session.entry.ledger.events()[-1].kind == "saved"
    assert not (session.entry.database_path / "runs").exists()


def test_save_all_writes_all_members_and_resaves_saved_artifacts(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session)
    first = session.save_artifacts()
    assert len(first.outputs) == 8  # data ×2, two figures ×2, two JSON
    assert all(
        item.status == "saved" and item.output.path.is_file() for item in first.outputs
    )
    sections = session.state.get_artifact_snapshots("tab")
    assert [section.section for section in sections] == [
        "run",
        "analysis",
        "post_analysis",
    ]
    assert all(
        artifact.status is SaveStatus.SAVED
        for section in sections
        for artifact in section.artifacts
    )
    primary_json = next(
        item.output.path
        for item in first.outputs
        if item.key.section == "analysis" and item.output.format == "json"
    )
    post_json = next(
        item.output.path
        for item in first.outputs
        if item.key.section == "post_analysis"
    )
    assert json.loads(primary_json.read_text())["parameters"] == {"gain": 2.0}
    assert json.loads(post_json.read_text())["summary"] == {"correction": 0.5}
    assert len(session.entry.ledger.events()) == 1
    saved_event = session.entry.ledger.events()[0].payload
    assert isinstance(saved_event, SavedPayload)
    assert len(saved_event.outputs) == 8
    second = session.save_artifacts()
    assert second.run_id == first.run_id
    assert second.outputs[0].output.path == first.outputs[0].output.path
    assert second.outputs[1].output.path != first.outputs[1].output.path
    metadata = require_record(session.state.get_tab("tab").run).metadata
    assert metadata is not None
    assert len(session.entry.ledger.events()) == 2


def test_json_only_save_uses_result_inputs_and_preserves_unselected_dirty_run(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    session.save()
    declare_panes(session, ())
    session.state.update_tab_analyze_param_instance("tab", PaneParams(99.0))
    session.control.set_comment("tab", "changed")
    completed = session.save_artifacts((ArtifactRef("analysis", "results"),))
    assert len(completed.outputs) == 1
    document = json.loads(completed.outputs[0].output.path.read_text())
    assert document["parameters"] == {"gain": 2.0}
    assert document["summary"] == {"mean": 3.0}
    sections = session.state.get_artifact_snapshots("tab")
    assert sections[0].artifacts[0].status is SaveStatus.UNSAVED_CHANGES
    assert sections[1].artifacts[0].status is SaveStatus.SAVED
    assert sections[2].artifacts[0].status is SaveStatus.NOT_SAVED


def test_save_encoding_and_io_complete_on_owner_with_member_failure_isolation(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure
    from zcu_tools.experiment.artifacts import saving as artifact_saving

    session = native_session
    plots = declare_panes(session)
    owner_id = threading.get_ident()
    encoded: list[Figure] = []
    io_threads: list[int] = []
    render = Figure.savefig
    native_write = artifact_saving.save

    def encode(figure: Figure, sink: BytesIO, *, format: str, dpi: float) -> None:
        assert threading.get_ident() == owner_id
        encoded.append(figure)
        if figure is plots["fit"]:
            raise RuntimeError("encoder failed")
        render(figure, sink, format=format, dpi=dpi)

    def write(
        destination: Path,
        container: Container,
        *,
        replace: bool = False,
    ) -> None:
        io_threads.append(threading.get_ident())
        native_write(destination, container, replace=replace)

    monkeypatch.setattr(Figure, "savefig", encode)
    monkeypatch.setattr(artifact_saving, "save", write)
    completed = session.save_artifacts()
    assert encoded == [plots["fit"], plots["residual"]]
    assert io_threads and all(thread == owner_id for thread in io_threads)
    failed = [item for item in completed.outputs if item.status == "failed"]
    assert len(failed) == 2
    assert all(
        item.key.member_name == "fit" and "encoder failed" in (item.error or "")
        for item in failed
    )
    assert all(
        item.output.path.is_file()
        for item in completed.outputs
        if item.status == "saved"
    )
    sections = session.state.get_artifact_snapshots("tab")
    assert sections[1].artifacts[0].status is SaveStatus.PARTIALLY_SAVED
    assert (
        sections[0].artifacts[0].status
        is sections[2].artifacts[0].status
        is SaveStatus.SAVED
    )
    assert len(session.entry.ledger.events()) == 1


def test_copy_path_failure_preserves_canonical_png_and_other_members(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session, ("fit",))
    planned = (
        session.state.get_artifact_snapshots("tab")[1].artifacts[0].members[0].outputs
    )
    planned[1].path.mkdir(parents=True)
    completed = session.save_artifacts()
    figures = [item for item in completed.outputs if item.key.member == "figure"]
    assert [item.status for item in figures] == ["saved", "failed"]
    assert figures[0].output.path.is_file()
    assert figures[1].error
    assert (
        session.state.get_artifact_snapshots("tab")[1].artifacts[0].status
        is SaveStatus.PARTIALLY_SAVED
    )
    assert len(session.entry.ledger.events()) == 1


@pytest.mark.parametrize(
    "refs",
    [
        (),
        (ArtifactRef("missing", "results"),),
        (ArtifactRef("run", "measurement_data"),) * 2,
    ],
)
def test_invalid_artifact_selection_preserves_annotation_and_admission(
    native_session: NativeSaveSession, refs: tuple[ArtifactRef, ...]
) -> None:
    session = native_session
    previous = session.state.get_artifact_snapshots("tab")
    with pytest.raises(FailedPreconditionError):
        session.control.save_artifacts(
            "tab", artifacts=refs, comment="rejected", origin=GUI_ORIGIN
        )
    assert session.state.get_tab("tab").save.comment == ""
    assert session.state.get_artifact_snapshots("tab") == previous
    assert not session.state.is_tab_busy("tab")
    assert session.artifact_completions == []
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


@pytest.mark.parametrize("missing", ["metadata", "storage"])
@pytest.mark.parametrize("batch", [False, True])
def test_missing_evidence_rejects_comment_before_operation(
    native_session: NativeSaveSession, missing: str, batch: bool
) -> None:
    session = native_session
    if missing == "metadata":
        session.state.update_tab_result(
            "tab",
            replace(require_record(session.state.get_tab("tab").run), metadata=None),
        )
    else:
        session.state.get_tab("tab").run.storage = None
    command = session.control.save_artifacts if batch else session.control.save_data
    with pytest.raises(FailedPreconditionError):
        command("tab", comment="rejected", origin=GUI_ORIGIN)
    assert session.state.get_tab("tab").save.comment == ""
    assert not session.state.is_tab_busy("tab")
    assert not session.completions and not session.artifact_completions
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


def test_analysis_replacement_and_new_run_retire_saved_sections(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    declare_panes(session)
    completed = session.save_artifacts()
    assert completed is not None
    session.state.update_tab_post_analyze(
        "tab", PostSummary(0.8), None, post_analyze_params_instance=PaneParams(5.0)
    )
    sections = session.state.get_artifact_snapshots("tab")
    assert sections[1].artifacts[0].status is SaveStatus.SAVED
    assert sections[2].artifacts[0].status is SaveStatus.NOT_SAVED
    session.state.update_tab_analyze(
        "tab", PaneSummary(7.0), None, analyze_params_instance=PaneParams(8.0)
    )
    sections = session.state.get_artifact_snapshots("tab")
    assert [section.section for section in sections] == ["run", "analysis"]
    assert sections[0].artifacts[0].status is SaveStatus.SAVED
    assert [member.key.member for member in sections[1].artifacts[0].members] == [
        "analysis"
    ]
    session.state.clear_tab_results("tab")
    assert session.state.get_artifact_snapshots("tab") == ()


def test_resave_all_failed_replaces_previous_success_evidence(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zcu_tools.experiment.artifacts import saving as artifact_saving

    session = native_session
    first = session.save()
    native_path = first.outputs[0].output.path
    native_path.unlink()
    native_path.mkdir()

    def fail_labber(*args: object, **kwargs: object) -> None:
        raise OSError("Labber denied")

    monkeypatch.setattr(artifact_saving, "write_labber", fail_labber)
    second = session.save()
    assert [item.status for item in second.outputs] == ["failed", "failed"]
    sections = session.state.get_artifact_snapshots("tab")
    data = sections[0].artifacts[0]
    assert data.status is SaveStatus.NOT_SAVED
    assert [output.status for output in data.members[0].outputs] == ["failed", "failed"]
    assert data.members[0].outputs[1].path == second.outputs[1].output.path
    assert data.members[0].outputs[1].path != first.outputs[1].output.path
    metadata = require_record(session.state.get_tab("tab").run).metadata
    assert metadata is not None
    assert session.state.get_tab("tab").run.confirmed_run_id == first.run_id
    session.state.clear_tab_results("tab")
    assert session.state.get_tab("tab").run.confirmed_run_id is None


def test_initial_all_failed_save_does_not_confirm_run_identity(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    native_path = (
        session.entry.database_path
        / "runs"
        / f"saving_{session.metadata.run_id}"
        / "data.h5"
    )
    native_path.mkdir(parents=True)
    session.entry.database_path.mkdir(parents=True, exist_ok=True)
    (session.entry.database_path / "Labber").write_text("not a directory")
    completed = session.save()
    assert [item.status for item in completed.outputs] == ["failed", "failed"]
    assert session.state.get_tab("tab").run.confirmed_run_id is None


def test_unsupported_analysis_field_rejects_before_annotation_or_files(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    session.state.update_tab_analyze(
        "tab", {"unsupported": object()}, None, analyze_params_instance=PaneParams()
    )
    with pytest.raises(TypeError, match=r"summary\.unsupported.*object"):
        session.control.save_artifacts("tab", comment="rejected", origin=GUI_ORIGIN)
    assert session.state.get_tab("tab").save.comment == ""
    assert not session.state.is_tab_busy("tab")
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


def test_ledger_failure_raises_without_publishing_a_complete_report(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = native_session
    declare_panes(session, ())

    def fail_ledger(*args: object, **kwargs: object) -> None:
        raise OSError("ledger denied")

    monkeypatch.setattr(type(session.entry.ledger), "append", fail_ledger)
    with pytest.raises(OSError, match="ledger denied"):
        session.save_artifacts()
    assert session.entry.ledger.events() == ()
    assert all(
        artifact.status is SaveStatus.NOT_SAVED
        for section in session.state.get_artifact_snapshots("tab")
        for artifact in section.artifacts
    )


def test_tab_snapshot_reads_committed_sections_without_payload_or_path_planning(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = native_session
    declare_panes(session)
    expected = session.state.get_artifact_snapshots("tab")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("read model attempted capture or planning")

    monkeypatch.setattr(SaveLayout, "outputs", forbidden)
    monkeypatch.setattr(
        type(session.state.get_tab("tab").adapter), "analysis_item", forbidden
    )
    tab_service = TabService(
        session.state,
        registry=MagicMock(),
        writeback=MagicMock(),
        cfg_resources=MagicMock(),
        form_sources=MagicMock(),
        captured_parameters=MagicMock(),
    )
    actual = tab_service.get_snapshot("tab")
    assert actual.artifacts == expected
    assert actual.save is not None and actual.save.comment == ""
    assert actual.analysis is not None and actual.analysis.result_params == PaneParams(
        2.0
    )
    assert (
        actual.post_analysis is not None
        and actual.post_analysis.result_params == PaneParams(4.0)
    )
    assert session.state.get_artifact_snapshots("tab") == expected
