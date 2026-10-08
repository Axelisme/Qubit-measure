"""Synchronous native Save through real state, writers and ledger."""

import json
import logging
import threading
from collections.abc import Generator, Mapping
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtCore import QEventLoop, QTimer
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import (
    CfgSnapshot,
    ExperimentPayload,
    NativeExtensions,
    RunMetadata,
    SoftwareProvenance,
    load_run_data,
)
from zcu_tools.experiment.acquisition import new_run_id
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.run_saving import ArtifactRef
from zcu_tools.gui.app.measure.adapter import RunRequest, SessionEnv
from zcu_tools.gui.app.measure.artifact_tracker import SaveStatus
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import (
    AcquiredPayload,
    ResultEntry,
    RoleSpec,
    SavedPayload,
    SaveLayout,
)

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._native_save_support import (
    GUI_ORIGIN,
    NativeSaveSession,
    PaneParams,
    PaneSummary,
    PostSummary,
    SavingAdapter,
    declare_panes,
    native_save_session,
)
from tests.resources.entry.fakes import (
    register_fakes,
    registry_state,
    restore_registry,
)


class RoleSavingAdapter(SavingAdapter):
    """Use the existing offline saving backend with a declared driver alias."""

    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        """Bind driver to the birth focus; allow manual rebinding on the tab."""
        return {"driver": RoleSpec("fake/drive/*")}


@pytest.fixture
def native_session(tmp_path: Path, qapp: QApplication) -> Generator[NativeSaveSession]:
    with native_save_session(tmp_path, qapp) as session:
        yield session


def _acquire_to_terminal(
    session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
    *,
    origin: CommandOrigin = GUI_ORIGIN,
) -> tuple[int, RunFinishedPayload]:
    gate = OperationGate(session.bus)
    progress = ProgressService(DirectProgressTransport())
    run = RunService(
        session.state,
        OperationRunner(
            gate, session.handles, progress, session.background, session.bus
        ),
        session.bus,
        session.handles,
        MagicMock(),
        gate=gate,
        devices=lambda: {},
    )

    def provenance(*, context: RunContext) -> SoftwareProvenance:
        return session.metadata.provenance

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance", provenance
    )
    cfg = session.state.get_tab("tab").cfg
    permit = GuardService(session.state).acquire_run_permit(
        "tab", expected_revision=cfg.observe().ref.revision
    )
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    finished: list[RunFinishedPayload] = []

    def terminal(payload: RunFinishedPayload) -> None:
        finished.append(payload)
        loop.quit()

    subscription = session.bus.subscribe(RunFinishedPayload, terminal)
    try:
        token = run.start_run(permit, plots=Plots(NonPresentingHost()), origin=origin)
        timer.start(5000)
        loop.exec()
    finally:
        timer.stop()
        subscription.unsubscribe()
    assert len(finished) == 1
    assert not session.state.is_tab_busy("tab")
    return token, finished[0]


@pytest.mark.parametrize("frontend", ["gui", "mcp"])
def test_native_run_without_save_records_acquired_at_its_historical_entry(
    native_session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
    frontend: str,
    tmp_path: Path,
) -> None:
    session = native_session
    other = ResultEntry.create(
        "other", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    other.new_point("q")
    session.context.install_entry(other)
    ContextControlFacet(context=session.context).use_point(PointUseRequest("q"))
    origin = GUI_ORIGIN if frontend == "gui" else CommandOrigin("mcp")
    token, terminal = _acquire_to_terminal(session, monkeypatch, origin=origin)
    assert terminal.outcome == "finished"
    outcome = session.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "finished"
    tab = session.state.get_tab("tab")
    metadata = tab.run.metadata
    record = tab.run.result
    assert metadata is not None and record is not None and record.cfg is not None
    events = session.entry.ledger.events()
    assert len(events) == 1
    event = events[0]
    assert event.kind == "acquired" and event.origin == frontend
    assert event.entry_id == str(session.entry.entry_id)
    assert event.at == metadata.finished_at
    assert isinstance(event.payload, AcquiredPayload)
    assert event.payload.run_id == metadata.run_id
    assert event.payload.tab == "saving" and event.payload.tab_id == "tab"
    assert event.payload.experiment == metadata.experiment
    assert event.payload.cfg_summary == record.cfg.model_dump(mode="json")
    assert event.payload.point == "p"
    assert event.payload.roles == dict(metadata.snapshot.roles)
    assert other.ledger.events() == ()
    assert tab.run.confirmed_run_id is None
    assert not (session.entry.database_path / "runs").exists()
    assert session.completions == []
    session.save()
    session.save()
    assert [event.kind for event in session.entry.ledger.events()] == [
        "acquired",
        "saved",
        "saved",
    ]


@pytest.fixture
def role_controller(
    native_session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Generator[tuple[State, Controller, BaseEventBus]]:
    """Compose an offline role-aware application and restore its registry/Qt owners."""
    before = registry_state()
    background = native_session.background
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    registry = Registry()
    registry.register("role_saving", RoleSavingAdapter)
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
        lambda: background,
    )

    def provenance(*, context: RunContext) -> SoftwareProvenance:
        return native_session.metadata.provenance

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance",
        provenance,
    )
    controller = Controller(
        state,
        registry,
        None,
        bus,
        project_root=str(tmp_path),
        progress_transport=DirectProgressTransport(),
    )
    try:
        register_fakes()
        installed = registry_state()
        controller.setup_control.new_entry(EntryRequest("rebound"))
        context = controller.context_control
        context.new_point(PointCreateRequest("p", "birth point", GUI_ORIGIN))
        for name in ("Q1", "Q2"):
            context.add_component(name, "fake/drive/a", {"rate": 10.0})
        context.set_component_focus("Q1")
        yield state, controller, bus
        assert registry_state() == installed
    finally:
        assert background.quiesce()
        for tab_id in tuple(state.list_tab_ids()):
            controller.tab_control.close_tab(tab_id)
        restore_registry(before)


@pytest.mark.parametrize("frontend", ["gui", "mcp"])
def test_native_role_rebind_records_new_alias_and_retains_result_source(
    role_controller: tuple[State, Controller, BaseEventBus], frontend: str
) -> None:
    state, controller, bus = role_controller
    control = controller.tab_control
    tab_id = control.new_tab("role_saving")
    tab = state.get_tab(tab_id)
    birth = tab.parameters
    assert (
        birth is not None
        and birth.roles is not None
        and birth.roles.names == {"driver": "Q1"}
    )
    storage = birth.capture_storage()
    control.set_role(tab_id, "driver", "Q2")
    origin = GUI_ORIGIN if frontend == "gui" else CommandOrigin("mcp")
    finished: list[RunFinishedPayload] = []
    loop, timer = QEventLoop(), QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    def terminal(payload: RunFinishedPayload) -> None:
        finished.append(payload)
        loop.quit()

    subscription = bus.subscribe(RunFinishedPayload, terminal)
    try:
        controller.run_analyze_control.start_run(
            tab_id, tab.cfg.observe().ref, origin=origin
        )
        timer.start(5000)
        if not finished:
            loop.exec()
    finally:
        timer.stop()
        subscription.unsubscribe()
    assert len(finished) == 1 and finished[0].outcome == "finished"
    assert not state.is_tab_busy(tab_id)
    metadata, parameters = tab.run.metadata, tab.run.parameters
    assert (
        metadata is not None and parameters is not None and parameters.roles is not None
    )
    assert metadata.snapshot.roles == parameters.roles.names == {"driver": "Q2"}
    assert (metadata.snapshot.entry_id, metadata.snapshot.point) == (
        storage.entry_id,
        storage.point,
    )
    (event,) = storage.ledger.events()
    assert (event.kind, event.origin, event.entry_id) == (
        "acquired",
        frontend,
        str(storage.entry_id),
    )
    assert isinstance(event.payload, AcquiredPayload)
    assert event.payload.run_id == metadata.run_id
    assert event.payload.roles == {"driver": "Q2"}
    assert event.payload.point == "p"
    assert not (storage.database_path / "runs").exists()
    control.set_role(tab_id, "driver", "Q1")
    assert (
        tab.parameters is not None
        and tab.parameters.roles is not None
        and tab.parameters.roles.names == {"driver": "Q1"}
    )
    assert tab.run.metadata is metadata and tab.run.parameters is parameters
    assert tab.save_context is birth
    report = controller.save_control.save_data(tab_id, origin=origin)
    assert all(output.status == "saved" for output in report.outputs)
    stored = load_run_data(report.outputs[0].output.path)
    assert stored.metadata.snapshot == metadata.snapshot
    assert stored.metadata.run_id == event.payload.run_id
    assert [event.kind for event in storage.ledger.events()] == [
        "acquired",
        "saved",
    ]


@pytest.mark.parametrize("failure", [False, True])
def test_native_run_stop_records_usable_result_but_failure_does_not(
    native_session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
) -> None:
    session = native_session
    adapter = session.state.get_tab("tab").adapter
    original = adapter.run

    def acquire(
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RunRecord[SavingCfg, SavingResult]:
        if failure:
            raise RuntimeError("offline acquisition failed")
        record = original(req, raw_cfg, context=context)
        context.cancel_signal.set()
        return record

    monkeypatch.setattr(adapter, "run", acquire)
    token, terminal = _acquire_to_terminal(session, monkeypatch)
    outcome = session.handles.known_outcome(token)
    assert outcome is not None
    events = session.entry.ledger.events()
    if failure:
        assert terminal.outcome == outcome.status == "failed"
        assert events == ()
        assert session.state.get_tab("tab").run.result is None
    else:
        assert terminal.outcome == outcome.status == "cancelled"
        metadata = session.state.get_tab("tab").run.metadata
        assert metadata is not None and metadata.completion == "stopped"
        assert len(events) == 1 and isinstance(events[0].payload, AcquiredPayload)
        assert events[0].payload.run_id == metadata.run_id
    assert not (session.entry.database_path / "runs").exists()


def test_native_run_ledger_failure_is_visible_and_retains_result_for_save(
    native_session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = native_session
    blocked = session.entry.result_path / "records" / "ledger.jsonl"
    blocked.mkdir()
    token, terminal = _acquire_to_terminal(session, monkeypatch)
    assert terminal.outcome == "failed" and terminal.error_message
    assert str(blocked) in terminal.error_message
    outcome = session.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error == terminal.error_message
    tab = session.state.get_tab("tab")
    assert tab.run.result is not None
    metadata = tab.run.metadata
    assert metadata is not None
    assert tab.run.source_operation_id == token
    assert tab.run.confirmed_run_id is None
    assert any(
        record.name == "zcu_tools.gui.app.measure.services.run"
        and record.levelno >= logging.ERROR
        for record in caplog.records
    )
    assert not (session.entry.database_path / "runs").exists()
    blocked.rmdir()
    saved = session.save()
    assert all(output.status == "saved" for output in saved.outputs)
    assert saved.run_id == metadata.run_id
    stored = load_run_data(saved.outputs[0].output.path)
    assert stored.metadata.started_at == metadata.started_at
    assert stored.metadata.finished_at == metadata.finished_at
    assert [event.kind for event in session.entry.ledger.events()] == ["saved"]


def test_native_run_without_storage_reports_missing_acquired_destination(
    native_session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = native_session
    session.state.get_tab("tab").save_context = None
    token, terminal = _acquire_to_terminal(session, monkeypatch)
    assert terminal.outcome == "finished"
    outcome = session.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "finished"
    assert session.state.get_tab("tab").run.result is not None
    assert session.state.get_tab("tab").run.metadata is None
    assert session.entry.ledger.events() == ()
    assert any(
        record.name == "zcu_tools.gui.app.measure.services.run"
        and record.levelno >= logging.WARNING
        for record in caplog.records
    )


def test_data_save_writes_both_formats_and_resave_keeps_run_identity(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    assert session.state.get_tab("tab").run.confirmed_run_id is None
    first = session.save()
    assert session.state.get_tab("tab").run.confirmed_run_id == first.run_id
    native, labber = first.outputs
    assert native.status == labber.status == "saved"
    assert (
        native.output.path
        == session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
    )
    stored = load_run_data(native.output.path)
    assert stored.metadata.run_id == session.metadata.run_id
    assert stored.metadata.started_at == session.metadata.started_at
    assert stored.metadata.snapshot == session.metadata.snapshot
    variable = next(iter(stored.payload.variables.values()))
    np.testing.assert_allclose(variable.axes[0].values, [1e9, 1.001e9, 1.002e9])
    second = session.save(origin=CommandOrigin("mcp"))
    assert second.run_id == first.run_id
    assert second.outputs[0].output.path == native.output.path
    assert second.outputs[1].output.path != labber.output.path
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
            session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
        )
        path.mkdir(parents=True)
    else:
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
        session.state.update_tab_result("tab", session.state.get_tab("tab").run.result)
    else:
        session.state.get_tab("tab").save_context = None
    with pytest.raises(FailedPreconditionError):
        session.control.save_data("tab", origin=GUI_ORIGIN)
    assert not session.state.is_tab_busy("tab")
    assert session.completions == []
    assert session.entry.ledger.events() == ()
    assert not (session.entry.database_path / "runs").exists()


def test_cancel_during_metadata_preserves_completed_acquisition_when_saved(
    native_session: NativeSaveSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = native_session
    owner = QtOwnerScheduler()
    gate = OperationGate(session.bus)
    progress = ProgressService(DirectProgressTransport())
    run = RunService(
        session.state,
        OperationRunner(
            gate, session.handles, progress, session.background, session.bus
        ),
        session.bus,
        session.handles,
        MagicMock(),
        gate=gate,
        devices=lambda: {},
    )

    def historical_provenance(*, context: RunContext) -> SoftwareProvenance:
        # Host/Git evidence is not the target of the Run -> Save contract.
        return session.metadata.provenance

    def identity_after_cancel(started_at: datetime) -> str:
        # This collaborator runs after the offline acquisition has returned.
        assert owner.call(run.cancel_run)
        return new_run_id(started_at)

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance",
        historical_provenance,
    )
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.new_run_id", identity_after_cancel
    )
    cfg = session.state.get_tab("tab").cfg
    permit = GuardService(session.state).acquire_run_permit(
        "tab", expected_revision=cfg.observe().ref.revision
    )
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    finished: list[RunFinishedPayload] = []

    def terminal(payload: RunFinishedPayload) -> None:
        finished.append(payload)
        loop.quit()

    subscription = session.bus.subscribe(RunFinishedPayload, terminal)
    try:
        token = run.start_run(permit, plots=Plots(NonPresentingHost()))
        timer.start(5000)
        loop.exec()
    finally:
        timer.stop()
        subscription.unsubscribe()
    assert len(finished) == 1 and finished[0].outcome == "cancelled"
    outcome = session.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "cancelled"
    metadata = session.state.get_tab("tab").run.metadata
    assert metadata is not None and metadata.completion == "complete"
    session.metadata = metadata
    saved = session.save()
    stored = load_run_data(saved.outputs[0].output.path)
    assert stored.metadata.completion == "complete"
    assert stored.metadata.run_id == metadata.run_id
    assert stored.metadata.started_at == metadata.started_at
    assert stored.metadata.finished_at == metadata.finished_at


def test_invalid_record_raises_without_writing_or_saved_event(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    session.state.update_tab_result("tab", object(), metadata=session.metadata)
    with pytest.raises(TypeError, match="RunRecord"):
        session.save()
    assert session.entry.ledger.events() == ()
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
    metadata = session.state.get_tab("tab").run.metadata
    assert metadata is not None
    assert metadata.labber_path == str(second.outputs[1].output.path)
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
    from zcu_tools.experiment import run_saving
    from zcu_tools.gui.app.measure.services import save as saving

    session = native_session
    plots = declare_panes(session)
    owner_id = threading.get_ident()
    encoded: list[Figure] = []
    io_threads: list[int] = []
    render = saving.render_save_figure_png
    native_write = run_saving.save_run_data

    def encode(figure: Figure) -> bytes:
        assert threading.get_ident() == owner_id
        encoded.append(figure)
        if figure is plots["fit"]:
            raise RuntimeError("encoder failed")
        return render(figure)

    def write(
        destination: Path,
        payload: ExperimentPayload,
        metadata: RunMetadata,
        *,
        cfg: CfgSnapshot,
        replace: bool = False,
        extensions: NativeExtensions | None = None,
    ) -> None:
        io_threads.append(threading.get_ident())
        native_write(
            destination,
            payload,
            metadata,
            cfg=cfg,
            replace=replace,
            extensions=extensions,
        )

    monkeypatch.setattr(saving, "render_save_figure_png", encode)
    monkeypatch.setattr(run_saving, "save_run_data", write)
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
        session.state.update_tab_result("tab", session.state.get_tab("tab").run.result)
    else:
        session.state.get_tab("tab").save_context = None
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
    from zcu_tools.experiment import run_saving

    session = native_session
    first = session.save()
    native_path = first.outputs[0].output.path
    native_path.unlink()
    native_path.mkdir()

    def fail_labber(*args: object, **kwargs: object) -> None:
        raise OSError("Labber denied")

    monkeypatch.setattr(run_saving, "write_labber", fail_labber)
    second = session.save()
    assert [item.status for item in second.outputs] == ["failed", "failed"]
    sections = session.state.get_artifact_snapshots("tab")
    data = sections[0].artifacts[0]
    assert data.status is SaveStatus.NOT_SAVED
    assert [output.status for output in data.members[0].outputs] == ["failed", "failed"]
    assert data.members[0].outputs[1].path == second.outputs[1].output.path
    assert data.members[0].outputs[1].path != first.outputs[1].output.path
    metadata = session.state.get_tab("tab").run.metadata
    assert metadata is not None
    assert metadata.labber_path == str(first.outputs[1].output.path)
    assert session.state.get_tab("tab").run.confirmed_run_id == first.run_id
    session.state.clear_tab_results("tab")
    assert session.state.get_tab("tab").run.confirmed_run_id is None


def test_initial_all_failed_save_does_not_confirm_run_identity(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    native_path = (
        session.entry.database_path / "runs" / session.metadata.run_id / "data.h5"
    )
    native_path.mkdir(parents=True)
    (session.entry.database_path / "Labber").write_text("not a directory")
    completed = session.save()
    assert [item.status for item in completed.outputs] == ["failed", "failed"]
    assert session.state.get_tab("tab").run.confirmed_run_id is None


def test_invalid_analysis_json_rejects_before_annotation_or_files(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    session.state.update_tab_analyze(
        "tab", PaneSummary(float("nan")), None, analyze_params_instance=PaneParams()
    )
    with pytest.raises(ValueError, match="strict JSON"):
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
        session.state.get_tab("tab").adapter, "capture_analysis_document", forbidden
    )
    tab_service = TabService(
        session.state,
        registry=MagicMock(),
        writeback=MagicMock(),
        cfg_resources=MagicMock(),
        cfg_resolution=MagicMock(),
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
