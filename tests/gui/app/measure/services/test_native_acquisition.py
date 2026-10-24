"""Native run evidence publication, acquired ledger events and retained admission source."""

import logging
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import SoftwareProvenance
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import (
    AppServices,
    build_app_services,
)
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import (
    AcquiredPayload,
    ResultEntry,
)

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._native_save_support import (
    GUI_ORIGIN,
    NativeSaveSession,
)
from tests.gui.app.measure._record_support import require_record
from tests.gui.app.measure._synthetic import ALIAS_PRESET
from tests.resources.entry.fakes import (
    register_fakes,
    registry_state,
    restore_registry,
)

from ._native_acquisition_support import RoleSavingAdapter


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
        refresh_devices=lambda: None,
    )

    def provenance(*, context: RunContext) -> SoftwareProvenance:
        return session.metadata.provenance

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance", provenance
    )
    session.state.get_tab("tab").cfg.reset()
    inputs = GuardService(session.state).capture_run_inputs("tab")
    finished: list[RunFinishedPayload] = []

    def terminal(payload: RunFinishedPayload) -> None:
        finished.append(payload)

    subscription = session.bus.subscribe(RunFinishedPayload, terminal)
    try:
        token = run.start_run(inputs, plots=Plots(NonPresentingHost()), origin=origin)
        # The admitted offline run returns and publishes its terminal payload.
        qt_until(lambda: bool(finished), label="native acquisition terminal")
    finally:
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
    metadata = require_record(tab.run).metadata
    record = require_record(tab.run)
    assert metadata is not None and record is not None and record.cfg is not None
    events = other.ledger.events()
    assert len(events) == 1
    event = events[0]
    assert event.kind == "acquired" and event.origin == frontend
    assert event.entry_id == str(other.entry_id)
    assert event.at == metadata.finished_at
    assert isinstance(event.payload, AcquiredPayload)
    assert event.payload.run_id == metadata.run_id
    assert event.payload.tab == "saving" and event.payload.tab_id == "tab"
    assert event.payload.experiment == metadata.experiment
    assert event.payload.point == "q"
    assert event.payload.roles == dict(metadata.snapshot.roles)
    assert session.entry.ledger.events() == ()
    assert tab.run.confirmed_run_id is None
    assert not (other.database_path / "runs").exists()
    assert session.completions == []
    session.save()
    session.save()
    assert [event.kind for event in other.ledger.events()] == [
        "acquired",
        "saved",
        "saved",
    ]


@pytest.fixture
def role_application(
    native_session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qapp: QApplication,
) -> Generator[tuple[State, AppServices, BaseEventBus]]:
    """Compose offline production services and restore registry/Qt owners."""
    before = registry_state()
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    registry = Registry()
    registry.register("role_saving", RoleSavingAdapter)

    def provenance(*, context: RunContext) -> SoftwareProvenance:
        return native_session.metadata.provenance

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.run.capture_software_provenance", provenance
    )
    services = build_app_services(
        alias_presets={"default": ALIAS_PRESET},
        state=state,
        bus=bus,
        registry=registry,
        progress_transport=DirectProgressTransport(),
        resource_versions=state.version.snapshot,
        render_host=lambda: None,
        owner_scheduler=QtOwnerScheduler(),
        project_root=str(tmp_path),
    )
    try:
        register_fakes()
        installed = registry_state()
        services.setup_control.new_entry(EntryRequest("birth"))
        services.context.new_point(PointCreateRequest("p", "birth point", GUI_ORIGIN))
        for name in ("Q1", "Q2"):
            services.context.add_component(name, "fake/drive/a", {"rate": 10.0})
        services.context.set_alias_rows((AliasRow("driver", "Q1"),))
        yield state, services, bus
        assert registry_state() == installed
    finally:
        for tab_id in tuple(state.list_tab_ids()):
            services.tab.close_tab(tab_id)
        assert services.background.quiesce()
        services.background.deleteLater()
        qapp.processEvents()
        restore_registry(before)


@pytest.mark.parametrize("frontend", ["gui", "mcp"])
@pytest.mark.parametrize("target", [None, "Q2"])
def test_run_save_retains_admission_source_after_global_entry_and_alias_changes(
    role_application: tuple[State, AppServices, BaseEventBus],
    frontend: str,
    target: str | None,
) -> None:
    state, services, bus = role_application
    tab_id = services.tab.new_tab("role_saving")
    tab = state.get_tab(tab_id)
    birth = services.context.read_context()
    birth_storage = birth.capture_storage()
    services.setup_control.new_entry(EntryRequest("acquisition"))
    services.context.new_point(
        PointCreateRequest("run-point", "Run source", GUI_ORIGIN)
    )
    for name in ("Q1", "Q2"):
        services.context.add_component(name, "fake/drive/a", {"rate": 20.0})
    services.context.set_alias_rows(
        (
            AliasRow("driver", "Q2" if target is None else "Q1"),
            AliasRow("custom", "Q1"),
        )
    )
    origin = GUI_ORIGIN if frontend == "gui" else CommandOrigin("mcp")
    finished: list[RunFinishedPayload] = []

    def terminal(payload: RunFinishedPayload) -> None:
        finished.append(payload)

    subscription = bus.subscribe(RunFinishedPayload, terminal)
    try:
        tab.cfg.reset(aliases=None if target is None else {"driver": target})
        services.run_analyze_control.start_run(tab_id, origin=origin)
        # The admitted role-bound run returns and publishes its terminal payload.
        qt_until(lambda: bool(finished), label="role acquisition terminal")
    finally:
        subscription.unsubscribe()
    assert len(finished) == 1 and finished[0].outcome == "finished"
    metadata, parameters = require_record(tab.run).metadata, tab.run.parameters
    assert metadata is not None and parameters is not None
    storage = parameters.capture_storage()
    assert metadata.snapshot.roles == {"driver": "Q2", "custom": "Q1"}
    assert (metadata.snapshot.entry_id, metadata.snapshot.point) == (
        storage.entry_id,
        "run-point",
    )
    (acquired,) = storage.ledger.events()
    assert isinstance(acquired.payload, AcquiredPayload)
    assert acquired.payload.run_id == metadata.run_id
    planned = state.get_artifact_snapshots(tab_id)
    assert all(
        output.path.is_relative_to(storage.database_path)
        for section in planned
        for artifact in section.artifacts
        for member in artifact.members
        for output in member.outputs
    )
    services.context.set_parameter(parameters.target, "Q2.rate", 30.0)
    services.context.set_alias_rows((AliasRow("driver", "Q1"),))
    services.setup_control.new_entry(EntryRequest("later"))
    assert parameters.value("$driver.rate") == 20.0
    report = services.save_control.save_data(tab_id, origin=origin)
    assert all(output.status == "saved" for output in report.outputs)
    stored = load(
        next(
            output.output.path
            for output in report.outputs
            if output.output.format == "data_h5"
        )
    )
    assert metadata_from_container(stored).snapshot == metadata.snapshot
    assert metadata_from_container(stored).run_id == acquired.payload.run_id
    assert [event.kind for event in storage.ledger.events()] == ["acquired", "saved"]
    assert not birth_storage.ledger.events()


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
        cfg: SavingCfg,
        *,
        context: RunContext,
    ) -> RunRecord[SavingCfg, SavingResult]:
        if failure:
            raise RuntimeError("offline acquisition failed")
        record = original(req, cfg, context=context)
        context.scope.stop.set()
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
        metadata = require_record(session.state.get_tab("tab").run).metadata
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
    assert require_record(tab.run) is not None
    metadata = require_record(tab.run).metadata
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
    stored = load(saved.outputs[0].output.path)
    assert metadata_from_container(stored).started_at == metadata.started_at
    assert metadata_from_container(stored).finished_at == metadata.finished_at
    assert [event.kind for event in session.entry.ledger.events()] == ["saved"]


def test_native_run_without_storage_reports_missing_acquired_destination(
    native_session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = native_session
    captured = session.state.session_env.parameters
    assert captured is not None
    session.state.set_context(
        replace(
            session.state.session_env,
            parameters=ParameterContext(
                point=captured.document(),
                library=captured.library,
                target=captured.target,
                roles=None,
            ),
        )
    )
    token, terminal = _acquire_to_terminal(session, monkeypatch)
    assert terminal.outcome == "finished"
    outcome = session.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "finished"
    assert session.state.get_tab("tab").run.result is not None
    assert require_record(session.state.get_tab("tab").run).metadata is None
    assert session.entry.ledger.events() == ()
    assert any(
        record.name == "zcu_tools.gui.app.measure.services.run"
        and record.levelno >= logging.WARNING
        for record in caplog.records
    )
