"""Native Load retains history and rejects unsafe saves through public services."""

from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import load
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.services.run_analyze_control import (
    RunAnalyzeControlFacet,
)
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.writeback import (
    WritebackControlFacet,
    WritebackDraft,
    WritebackEdit,
    WritebackService,
    WritebackWritten,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import (
    AcceptedPayload,
    ResultEntry,
    RoleResolutionError,
    SavedPayload,
)

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui.app.measure._form_fakes import form_resources, form_sources
from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    PaneParams,
    PaneSummary,
    PostSummary,
    declare_panes,
    native_save_session,
)
from tests.gui.app.measure._record_support import require_metadata, require_record


def _native_controls(
    session: NativeSaveSession,
    writeback: WritebackService,
    control: ContextControlFacet,
) -> tuple[RunAnalyzeControlFacet, WritebackControlFacet]:
    """Compose real admission/load/analysis/WB owners; no Run is executed."""

    def capture(aliases: Mapping[str, str] | None) -> ParameterContext | None:
        return session.capture_parameters(aliases)

    guard = GuardService(session.state)
    runner = OperationRunner(
        None, session.handles, None, session.background, session.bus
    )
    tab = TabService(
        session.state,
        MagicMock(),
        writeback,
        form_resources(session.state),
        lambda aliases: form_sources(session.state, aliases),
        capture,
    )
    load = LoadService(
        session.state,
        writeback,
        context=control,
        captured_parameters=capture,
    )
    run = RunService(
        session.state,
        runner,
        session.bus,
        session.handles,
        writeback,
        gate=MagicMock(),
        devices=lambda: {},
        refresh_devices=lambda: None,
    )
    analyze = AnalyzeService(
        session.state, runner, session.bus, writeback, session.handles
    )
    post = PostAnalyzeService(
        session.state, runner, session.bus, session.handles, writeback
    )
    commands = RunAnalyzeControlFacet(
        state=session.state,
        bus=session.bus,
        guard=guard,
        tab=tab,
        load=load,
        run=run,
        analyze=analyze,
        post_analyze=post,
        render_host=lambda: None,
        owner_scheduler=QtOwnerScheduler(),
    )
    wb = WritebackControlFacet(
        state=session.state,
        guard=guard,
        writeback=writeback,
        resource_versions=session.state.version.snapshot,
        bus=session.bus,
    )
    return commands, wb


@pytest.fixture
def role_load_session(
    tmp_path: Path,
    qapp: QApplication,
    result_sources: tuple[ParameterContext, ParameterContext],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[NativeSaveSession]:
    """Save Q1 roles at p, then update persisted Q1.rate before native Load."""
    del result_sources  # Fixture owns the component-kind registration lifecycle.
    with native_save_session(tmp_path, qapp) as session:
        control = ContextControlFacet(context=session.context)
        tab = session.state.get_tab("tab")
        control.add_component("Q1", "run-source/qubit", {"rate": 1.0})
        control.add_component("Q2", "run-source/qubit", {"rate": 2.0})
        session.context.set_alias_rows(
            (
                AliasRow("source_qubit", "Q1"),
                AliasRow("custom", "Q2"),
            )
        )
        captured = session.context.read_context()
        assert captured is not None and captured.roles is not None
        session.metadata = session.metadata.model_copy(
            update={
                "snapshot": replace(
                    session.metadata.snapshot, roles=captured.roles.names
                )
            }
        )
        session.state.update_tab_result(
            "tab",
            replace(require_record(tab.run), metadata=session.metadata),
            parameters=captured,
        )
        session.save()
        control.set_parameter(captured.target, "Q1.rate", 7.0)
        yield session


def test_native_load_analysis_and_writeback_keep_historical_point_aliases_and_id(
    role_load_session: NativeSaveSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = role_load_session
    control = ContextControlFacet(context=session.context)
    tab = session.state.get_tab("tab")
    target = control.read_context().target
    path = session.native_data_path
    session.entry.new_point("other")
    other = control.use_point(PointUseRequest("other"))
    control.add_component("Q1", "run-source/qubit", {"rate": 40.0})
    control.add_component("Q2", "run-source/qubit", {"rate": 50.0})
    session.context.set_alias_rows(
        (
            AliasRow("source_qubit", "Q2"),
            AliasRow("custom", "Q1"),
        )
    )
    rows = session.context.get_alias_rows()
    service = LoadService(
        session.state,
        MagicMock(),
        context=control,
        captured_parameters=session.capture_parameters,
    )

    service.load_result("tab", str(path))

    loaded = tab.run.parameters
    assert loaded is not None
    assert loaded.target == target
    assert loaded.roles is not None and loaded.roles.names == {
        "source_qubit": "Q1",
        "custom": "Q2",
    }
    assert tab.run.storage is not None and tab.run.storage.point == "other"
    assert control.read_context().target == other
    control.set_parameter(target, "Q1.rate", 10.0)
    seen: list[AnalyzeRequest[object, object]] = []

    def analyze(
        request: AnalyzeRequest[object, object], *, plots: Plots
    ) -> PaneSummary:
        del plots
        seen.append(request)
        return PaneSummary()

    def proposals(
        request: WritebackRequest[object, PaneSummary],
    ) -> list[WritebackItem]:
        assert request.ctx.parameters is loaded
        return [ParameterWriteback("$source_qubit.rate", "Rate", proposed_value=12.0)]

    monkeypatch.setattr(tab.adapter, "analyze", analyze)
    monkeypatch.setattr(tab.adapter, "get_writeback_items", proposals)
    writeback = WritebackService(control, arb_names=lambda: ())
    runner = OperationRunner(
        None, session.handles, None, session.background, session.bus
    )
    analysis = AnalyzeService(
        session.state, runner, session.bus, writeback, session.handles
    )
    analysis.start_analyze("tab", PaneParams(), plots=Plots(NonPresentingHost()))
    assert session.background.quiesce()
    assert len(seen) == 1 and seen[0].parameters is loaded
    assert loaded.value("$source_qubit.rate") == 7.0
    draft = tab.analysis.writeback_draft
    assert isinstance(draft, WritebackDraft)
    assert draft.parameters is loaded
    assert draft.run_id == session.metadata.run_id
    assert writeback.preview_values(draft)["param-1"].current == 10.0
    assert writeback.write_draft(
        draft,
        (WritebackEdit("param-1"),),
        origin=CommandOrigin("mcp"),
    ) == [WritebackWritten("param-1", "parameter", "Q1.rate", 10.0, 12.0)]
    event = session.entry.ledger.events()[-1]
    assert (
        isinstance(event.payload, AcceptedPayload)
        and event.payload.run_id == session.metadata.run_id
    )
    assert control.read_context().target == other
    assert control.read_context().value("Q2.rate") == 50.0
    assert control.read_target(target).value("Q1.rate") == 12.0
    assert session.context.get_alias_rows() == rows
    writeback.teardown_draft(draft)


@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_native_load_missing_source_is_reported_but_corruption_preserves_previous_result(
    role_load_session: NativeSaveSession,
    fault: str,
) -> None:
    session = role_load_session
    tab = session.state.get_tab("tab")
    history = session.metadata.snapshot
    broken = replace(history, point="missing" if fault == "missing" else history.point)
    session.state.update_tab_result(
        "tab",
        replace(
            require_record(tab.run),
            metadata=session.metadata.model_copy(update={"snapshot": broken}),
        ),
        parameters=tab.run.parameters,
    )
    session.save()
    previous = tab.run
    events = session.entry.ledger.events()
    service = LoadService(
        session.state,
        MagicMock(),
        context=ContextControlFacet(context=session.context),
        captured_parameters=session.capture_parameters,
    )
    if fault == "corrupt":
        source = (
            session.entry.result_path / "points" / str(history.point) / "point.yaml"
        )
        source.write_text("format: invalid\\n", encoding="utf-8")
        with pytest.raises(ValueError, match="format"):
            service.load_result("tab", str(session.native_data_path))
        assert tab.run is previous
    else:
        outcome = service.load_result("tab", str(session.native_data_path))
        assert "Historical point 'missing' is unavailable" in str(
            outcome.analysis_source_issue
        )
        assert tab.run.analysis_source_issue == outcome.analysis_source_issue
        assert tab.run.parameters is None
        assert require_metadata(tab.run).snapshot == broken
    assert session.entry.ledger.events() == events


def test_native_load_keeps_unknown_historical_name_until_component_field_read(
    role_load_session: NativeSaveSession,
) -> None:
    session = role_load_session
    tab = session.state.get_tab("tab")
    previous = tab.run
    history = replace(session.metadata.snapshot, roles={"source_qubit": "missing"})
    metadata = session.metadata.model_copy(update={"snapshot": history})
    session.state.update_tab_result(
        "tab",
        replace(require_record(previous), metadata=metadata),
        parameters=previous.parameters,
    )
    session.save()
    saved_metadata = require_record(tab.run).metadata
    events = session.entry.ledger.events()
    rows = session.context.get_alias_rows()
    service = LoadService(
        session.state,
        MagicMock(),
        context=ContextControlFacet(context=session.context),
        captured_parameters=session.capture_parameters,
    )

    service.load_result(
        "tab",
        str(session.native_data_path),
    )

    loaded = tab.run.parameters
    assert loaded is not None and loaded is not previous.parameters
    assert require_record(tab.run).metadata == saved_metadata
    assert loaded.roles is not None and loaded.roles.names == history.roles
    assert loaded.value("$source_qubit") == "missing"
    with pytest.raises(RoleResolutionError) as error:
        loaded.value("$source_qubit.rate")
    assert error.value.role == "missing"
    assert previous.parameters is not None
    assert previous.parameters.value("$source_qubit.rate") == 1.0
    assert loaded.value("Q1.rate") == 7.0
    assert session.context.get_alias_rows() == rows
    assert session.entry.ledger.events() == events


def assert_empty_writeback_admission(
    wb: WritebackControlFacet, *, selected: bool
) -> None:
    """An empty proposal still needs selection at the application boundary."""
    if selected:
        assert wb.apply_writeback_for_pane("tab", "analysis")["applied_ids"] == []
    else:
        with pytest.raises(FailedPreconditionError) as caught:
            wb.apply_writeback_for_pane("tab", "analysis")
        assert caught.value.reason_code == "no_context"


@pytest.mark.parametrize("source", ["foreign", "empty", "pointless"])
def test_native_load_without_readable_source_allows_context_free_analysis(
    role_load_session: NativeSaveSession,
    tmp_path: Path,
    source: str,
) -> None:
    session = role_load_session
    tab = session.state.get_tab("tab")
    path = session.native_data_path
    if source == "foreign":
        other = ResultEntry.create(
            "foreign",
            result_root=tmp_path / "Foreign",
            database_root=tmp_path / "ForeignDB",
        )
        other.new_point("foreign-point")
        session.context.install_entry(other)
        ContextControlFacet(context=session.context).use_point(
            PointUseRequest("foreign-point")
        )
    elif source == "empty":
        session.context.clear_context()
    else:
        session.state.update_tab_result(
            "tab",
            replace(
                require_record(tab.run),
                metadata=session.metadata.model_copy(
                    update={"snapshot": replace(session.metadata.snapshot, point=None)}
                ),
            ),
            parameters=tab.run.parameters,
        )
        session.save()
    control = ContextControlFacet(context=session.context)
    writeback = WritebackService(control, arb_names=lambda: ())
    commands, wb = _native_controls(session, writeback, control)
    loaded = commands.load_tab_result("tab", str(path))
    assert loaded.has_analyze_params
    assert loaded.analysis_source_issue == {
        "foreign": "Historical entry differs from the opened entry",
        "empty": "No active context is available",
        "pointless": "Historical metadata has no point",
    }[source]
    assert tab.run.analysis_source_issue == loaded.analysis_source_issue
    assert require_record(tab.run).metadata is not None
    assert require_metadata(tab.run).run_id == session.metadata.run_id
    assert tab.run.parameters is None
    if source == "empty":
        assert tab.run.storage is None
    commands.analyze("tab", PaneParams())
    assert session.background.quiesce()
    assert isinstance(tab.analysis.result, PaneSummary)
    draft = tab.analysis.writeback_draft
    assert isinstance(draft, WritebackDraft)
    assert draft.parameters is None and draft.run_id == session.metadata.run_id
    assert draft.items == ()
    events = session.entry.ledger.events()
    assert_empty_writeback_admission(wb, selected=source != "empty")
    assert session.entry.ledger.events() == events
    writeback.teardown_draft(draft)
    rejected = writeback.create_draft(
        [ParameterWriteback("Q1.rate", "rate", 99.0)],
        parameters=None,
        run_id=session.metadata.run_id,
        experiment_name="test/writeback",
    )
    tab.analysis.writeback_draft = rejected
    selection = control.get_source_binding()
    try:
        message = "working point" if source == "empty" else "captured context"
        with pytest.raises(FailedPreconditionError, match=message):
            wb.apply_writeback_for_pane("tab", "analysis")
        assert session.entry.ledger.events() == events
        assert control.get_source_binding() == selection
        assert rejected.is_active
        assert not any(wb.get_writeback_applied_for_pane("tab", "analysis").values())
    finally:
        writeback.teardown_draft(rejected)


def test_public_writeback_rejects_foreign_entry_without_acceptance_or_write(
    role_load_session: NativeSaveSession, tmp_path: Path
) -> None:
    session = role_load_session
    control = ContextControlFacet(context=session.context)
    tab = session.state.get_tab("tab")
    parameters = tab.run.parameters
    assert parameters is not None and isinstance(parameters.target, SavedPoint)
    writeback = WritebackService(control, arb_names=lambda: ())
    _, wb = _native_controls(session, writeback, control)
    draft = writeback.create_draft(
        [ParameterWriteback("$source_qubit.rate", "rate", 99.0)],
        parameters=parameters,
        run_id=session.metadata.run_id,
        experiment_name="test/writeback",
    )
    session.state.update_tab_analyze("tab", PaneSummary(), None, writeback_draft=draft)
    other = ResultEntry.create(
        "foreign",
        result_root=tmp_path / "Foreign",
        database_root=tmp_path / "ForeignDB",
    )
    other.new_point("other")
    session.context.install_entry(other)
    control.use_point(PointUseRequest("other"))
    original_events = session.entry.ledger.events()
    foreign_events = other.ledger.events()
    selection = control.get_source_binding()
    try:
        with pytest.raises(FailedPreconditionError) as caught:
            wb.apply_writeback_for_pane("tab", "analysis")
        assert caught.value.reason_code == "entry_not_open"
        assert session.entry.ledger.events() == original_events
        assert other.ledger.events() == foreign_events
        assert control.get_source_binding() == selection
        assert draft.is_active
        assert not any(wb.get_writeback_applied_for_pane("tab", "analysis").values())
        session.context.install_entry(session.entry)
        control.use_point(PointUseRequest(parameters.target.point))
        assert control.read_context().lookup("Q1.rate").value == 7.0
    finally:
        writeback.teardown_draft(draft)


@pytest.mark.parametrize("started_at", [None, "2026-10-08T00:00:00+00:00"])
@pytest.mark.parametrize("completion", ["complete", "partial", "stopped", "unknown"])
def test_native_load_retains_history_and_resaves_captured_source(
    tmp_path: Path,
    qapp: QApplication,
    completion: Literal["complete", "partial", "stopped", "unknown"],
    started_at: str | None,
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.metadata = session.metadata.model_copy(
            update={"completion": completion, "started_at": started_at}
        )
        session.state.update_tab_result(
            "tab",
            replace(
                require_record(session.state.get_tab("tab").run),
                metadata=session.metadata,
            ),
            parameters=session.state.get_tab("tab").run.parameters,
        )
        initial_report = session.save()
        assert all(output.status == "saved" for output in initial_report.outputs), (
            "Historical unknown evidence must be saveable"
        )
        path = session.native_data_path
        stored = load(path)
        events = session.entry.ledger.events()
        service = LoadService(
            session.state,
            MagicMock(),
            context=ContextControlFacet(context=session.context),
            captured_parameters=session.capture_parameters,
        )
        outcome = service.load_result("tab", str(path))
        original_capture = session.state.get_tab("tab").run.parameters
        other = ResultEntry.create(
            "other",
            result_root=tmp_path / "Elsewhere",
            database_root=tmp_path / "ElseDB",
        )
        other.new_point("other-point")
        session.context.install_entry(other)
        ContextControlFacet(context=session.context).use_point(
            PointUseRequest("other-point")
        )
        tab = session.state.get_tab("tab")
        record = require_record(tab.run)
        assert isinstance(record, RunRecord)
        assert isinstance(record.cfg, SavingCfg)
        assert isinstance(record.result, SavingResult)
        np.testing.assert_array_equal(record.result.signal, [1 + 2j, 3 - 4j, 5 + 6j])
        assert outcome.result_type == "RunRecord"
        assert require_record(tab.run).metadata == metadata_from_container(stored)
        assert tab.run.source_path == str(path)
        assert tab.run.confirmed_run_id is None
        assert tab.run.parameters is original_capture
        assert session.entry.ledger.events() == events

        report = session.save()

        assert report.run_id == metadata_from_container(stored).run_id
        saved = metadata_from_container(load(path))
        assert saved == metadata_from_container(stored)
        assert all(event.kind == "saved" for event in session.entry.ledger.events())
        assert not other.ledger.events()


@pytest.mark.parametrize("save_all", [False, True])
def test_loaded_same_entry_different_point_resaves_original_run(
    tmp_path: Path, qapp: QApplication, save_all: bool
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        source = session.native_data_path
        stored = metadata_from_container(load(source))
        assert stored.snapshot.point == "p"
        session.entry.new_point("destination-point")
        control = ContextControlFacet(context=session.context)
        control.use_point(PointUseRequest("destination-point"))
        tab = session.state.get_tab("tab")
        service = LoadService(
            session.state,
            MagicMock(),
            context=ContextControlFacet(context=session.context),
            captured_parameters=session.capture_parameters,
        )
        service.load_result("tab", str(source))
        assert tab.run.confirmed_run_id is None
        planned_copies: tuple[Path, ...] = ()
        if save_all:
            declare_panes(session, ("fit",))
            post_plots = Plots(NonPresentingHost())
            post_plots.subplots("post")
            post_plots.finish(present=False)
            session.state.update_tab_post_analyze(
                "tab",
                PostSummary(),
                post_plots,
                post_analyze_params_instance=PaneParams(4.0),
            )
            planned_copies = tuple(
                output.path
                for section in session.state.get_artifact_snapshots("tab")
                for artifact in section.artifacts
                for member in artifact.members
                for output in member.outputs
                if output.format == "png"
                and output.path.is_relative_to(session.entry.result_path)
            )
            assert len(planned_copies) == 2
            assert all(
                path.parent
                == session.entry.result_path
                / "points"
                / stored.snapshot.point
                / "figures"
                for path in planned_copies
            )
        previous_events = session.entry.ledger.events()

        save = session.save_artifacts if save_all else session.save
        report = save()

        assert report.run_id == stored.run_id
        assert all(item.status == "saved" for item in report.outputs)
        assert (
            next(
                item.output.path
                for item in report.outputs
                if item.output.format == "data_h5"
            )
            == source
        )
        saved = metadata_from_container(load(source))
        assert require_record(tab.run).metadata == saved
        assert tab.run.confirmed_run_id == stored.run_id
        assert (
            tab.run.parameters is not None
            and tab.run.parameters.target
            == SavedPoint(
                entry_id=str(stored.snapshot.entry_id),
                point=stored.snapshot.point,
            )
        )
        assert (
            tab.run.storage is not None and tab.run.storage.point == "destination-point"
        )
        events = session.entry.ledger.events()
        assert events[:-1] == previous_events
        event = events[-1]
        assert event.kind == "saved"
        assert event.entry_id == str(stored.snapshot.entry_id)
        assert isinstance(event.payload, SavedPayload)
        assert event.payload.run_id == stored.run_id
        record = session.entry.ledger.get(event.id).record
        assert record is not None
        assert record["point"] == stored.snapshot.point
        assert list((session.entry.database_path / "runs").iterdir()) == [source.parent]
        if save_all:
            copies = [
                item.output.path
                for item in report.outputs
                if item.output.format == "png"
                and item.output.path.is_relative_to(session.entry.result_path)
            ]
            assert tuple(copies) == planned_copies
            assert all(path.is_file() for path in copies)


@pytest.mark.parametrize("save_all", [False, True])
def test_loaded_foreign_origin_save_rejects_before_mapping_or_io(
    tmp_path: Path,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    save_all: bool,
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        source = session.native_data_path
        destination = ResultEntry.create(
            "destination",
            result_root=tmp_path / "Destination",
            database_root=tmp_path / "DestinationDB",
        )
        destination.new_point("destination-point")
        session.context.install_entry(destination)
        control = ContextControlFacet(context=session.context)
        control.use_point(PointUseRequest("destination-point"))
        tab = session.state.get_tab("tab")
        captured = control.read_context().capture_storage()
        assert captured.point == "destination-point"
        service = LoadService(
            session.state,
            MagicMock(),
            context=ContextControlFacet(context=session.context),
            captured_parameters=session.capture_parameters,
        )
        service.load_result("tab", str(source))
        assert require_record(tab.run).metadata == metadata_from_container(load(source))
        assert tab.run.parameters is None
        assert tab.run.storage is not None
        assert tab.run.storage.entry_id == captured.entry_id
        before = {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }

        def reject_mapping(key: object, result: object, *, comment: str) -> None:
            raise AssertionError("Origin rejection must precede data mapping")

        monkeypatch.setattr(tab.adapter, "data_item", reject_mapping)
        save = session.save_artifacts if save_all else session.save
        with pytest.raises(FailedPreconditionError) as error:
            save()

        message = str(error.value)
        assert str(session.metadata.snapshot.entry_id) in message
        assert session.metadata.snapshot.point == "p"
        assert session.metadata.snapshot.point in message
        assert str(captured.entry_id) in message
        assert captured.point in message
        after = {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }
        assert after == before
        assert tab.run.confirmed_run_id is None


@pytest.mark.parametrize("save_all", [False, True])
def test_unselected_entry_load_retains_history_and_save_rejects_without_writes(
    tmp_path: Path, qapp: QApplication, save_all: bool
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        declare_panes(session)
        path = session.native_data_path
        stored = load(path)
        session.context.install_entry(session.entry)
        control = ContextControlFacet(context=session.context)
        assert session.state.session_env.parameters is None
        tab = session.state.get_tab("tab")
        before = {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }
        events = session.entry.ledger.events()
        service = LoadService(
            session.state,
            MagicMock(),
            context=ContextControlFacet(context=session.context),
            captured_parameters=session.capture_parameters,
        )

        outcome = service.load_result("tab", str(path))

        assert outcome.result_type == "RunRecord"
        assert isinstance(require_record(tab.run), RunRecord)
        metadata = require_record(tab.run).metadata
        assert metadata is not None and metadata == metadata_from_container(stored)
        assert tab.run.source_path == str(path)
        assert tab.run.confirmed_run_id is None
        assert tab.run.parameters is None
        assert str(metadata.snapshot.entry_id) == session.entry.entry_id
        assert metadata.snapshot.point == "p"
        assert tab.run.storage is None
        assert tab.analysis.result is None
        assert tab.post_analysis.result is None
        assert tab.artifacts.snapshot() == ()
        # Later active selection cannot fill this Load's missing storage.
        control.use_point(PointUseRequest("p"))
        save = session.save_artifacts if save_all else session.save
        with pytest.raises(FailedPreconditionError):
            save()
        assert session.entry.ledger.events() == events
        assert {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file()
        } == before


def test_native_experiment_mismatch_keeps_previous_result_and_history(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        tab = session.state.get_tab("tab")
        session.state.update_tab_result(
            "tab",
            replace(
                require_record(tab.run),
                metadata=session.metadata.model_copy(
                    update={"experiment": "test/different"}
                ),
            ),
            parameters=tab.run.parameters,
        )
        session.save()
        path = session.native_data_path
        old_run = tab.run
        service = LoadService(
            session.state,
            MagicMock(),
            context=ContextControlFacet(context=session.context),
            captured_parameters=session.capture_parameters,
        )

        with pytest.raises(LoadDataError) as error:
            service.load_result("tab", str(path))

        assert error.value.reason_code == "invalid_data_file"
        assert "test/different" in str(error.value)
        assert tab.run is old_run


def test_unreadable_native_load_keeps_previous_result_and_history(
    tmp_path: Path, qapp: QApplication
) -> None:
    with native_save_session(tmp_path, qapp) as session:
        tab = session.state.get_tab("tab")
        old_run = tab.run
        service = LoadService(
            session.state,
            MagicMock(),
            context=ContextControlFacet(context=session.context),
            captured_parameters=session.capture_parameters,
        )

        with pytest.raises(LoadDataError) as error:
            service.load_result("tab", str(tmp_path / "missing.h5"))

        assert error.value.reason_code == "data_file_read_failed"
        assert tab.run is old_run
        assert require_record(tab.run).metadata == session.metadata
