"""Entry selection admission at real application, view and wire seams."""

from collections.abc import Iterator

import pytest
from qtpy.QtWidgets import QLabel, QPushButton
from zcu_tools.gui.app.measure.adapter import ParameterWriteback
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.guard import GuardService, TabAction
from zcu_tools.gui.app.measure.services.pane_commit import commit_analysis
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.services.tab_control import TabControlFacet
from zcu_tools.gui.app.measure.services.workspace import WorkspaceService
from zcu_tools.gui.app.measure.state import AnalysisPaneState
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.writeback import WritebackControlFacet, WritebackService
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.remote.errors import ErrorCode
from zcu_tools.gui.session.context_types import EntryRequest, SavedEntry
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.types import ContextReadiness
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.resources.entry import ResultEntry

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._form_fakes import form_resources, form_sources
from tests.gui.app.measure._record_support import require_target

from ._point_application import PointAdapter, PointApplication


def entry_resource(application: PointApplication) -> ResultEntry:
    info = application.controller.context_control.get_entry_info()
    assert info is not None
    return ResultEntry.open(
        info.name,
        result_root=info.result_path.parent,
        database_root=info.database_path.parent,
    )


@pytest.mark.parametrize("command", ["new", "open"])
def test_entry_commands_activate_global_for_application_and_wire_reads(
    application: PointApplication, command: str
) -> None:
    if command == "open":
        application.open_entry()
        application.result("point.new", {"label": "existing"})
    application.result(f"entry.{command}", {"name": "entry"})
    assert application.state.session_env.readiness is ContextReadiness.GLOBAL
    assert application.state.session_env.active_label is None
    assert application.controller.context_control.get_entry_info() is not None
    labels = application.result("point.list", {})["points"]
    assert labels == (
        []
        if command == "new"
        else [
            {"label": "existing", "description": "", "available": True, "error": None}
        ]
    )
    control = application.controller.context_control
    assert control.read_context().target == SavedEntry(entry_name="entry")
    control.add_component("Q1", "fake/drive/a", {})
    snapshot = application.result("context.snapshot", {})
    assert snapshot["target"] == SavedEntry(entry_name="entry").model_dump()
    assert control.read_context().component("Q1").kind == "fake/drive/a"
    tab_id = application.controller.tab_control.new_tab("pointed")
    with pytest.raises(FailedPreconditionError) as caught:
        application.controller.run_analyze_control.start_run(tab_id)
    # Global satisfies context admission; the independent SoC prerequisite remains.
    assert caught.value.reason_code == "no_soc"
    reply = application.rpc("tab.run_start", {"tab_id": tab_id})
    assert reply["ok"] is False
    assert isinstance(reply["error"], dict)
    assert reply["error"]["code"] == ErrorCode.PRECONDITION_FAILED
    assert reply["error"]["reason"] == "no_soc"
    application.result(
        "point.new" if command == "new" else "point.use",
        {"label": "selected" if command == "new" else "existing"},
    )
    capture = control.read_context()
    control.set_parameter(require_target(capture), "general.ext.flux_value", 2.0)
    assert (
        control.read_target(require_target(capture)).value("general.ext.flux_value")
        == 2.0
    )


def test_global_allows_writeback_and_inspect_while_empty_requires_entry(
    application: PointApplication, qapp
) -> None:
    application.open_entry()
    tab_id = application.controller.tab_control.new_tab("pointed")
    commit_analysis(
        application.state,
        tab_id,
        AnalysisPaneState(
            result=object(),
            plots=None,
            params=application.state.get_tab(tab_id).analysis.params,
        ),
    )
    window = MainWindow(
        application.controller,
        templates=lambda: {},
        dialog_presenter=RecordingDialogPresenter(),
    )
    window.add_tab_widget(tab_id, "pointed")
    editor = ComponentTreeWidget(application.controller.context_control)
    try:
        run = next(
            button
            for button in window.findChildren(QPushButton)
            if button.text() == "Run"
        )
        assert not run.isEnabled()
        assert run.toolTip().strip()
        tab = window.findChild(ExpTabWidget)
        assert tab is not None
        guard = GuardService(application.state, access=ExperimentAccess())
        assert guard.admit(tab_id, TabAction.WRITEBACK).allowed
        # No draft is installed, independently of destination readiness.
        assert not tab.writeback_widget.isEnabled()
        # Global has active parameters even without a point.
        hint = editor.findChild(QLabel, "component_hint")
        assert hint is not None
        assert hint.isHidden()
        add = next(
            button
            for button in editor.findChildren(QPushButton)
            if button.text().startswith("Add component")
        )
        assert add.isEnabled()
        assert editor.edit_parameter("general.ext.flux_value", 1.0)
        application.controller.session_services.context.clear_context()
        editor.refresh()
        assert guard.admit(tab_id, TabAction.WRITEBACK).reason == "no_context"
        assert not tab.writeback_widget.isEnabled()
        assert tab.writeback_widget.toolTip().strip()
        assert not run.isEnabled()
        assert not hint.isHidden()
        assert hint.text().strip()
        assert not add.isEnabled()
        assert not editor.edit_parameter("general.ext.flux_value", 1.0)
        application.controller.setup_control.open_entry(EntryRequest("entry"))
        application.result("point.new", {"label": "selected"})
        editor.refresh()
        assert add.isEnabled()
        assert editor.edit_parameter("general.ext.flux_value", 1.0)
    finally:
        editor.deleteLater()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("frontend", ["application", "wire"])
def test_writeback_uses_global_after_reopen_then_current_selected_destination(
    application: PointApplication, frontend: str
) -> None:
    application.open_entry()
    application.result("point.new", {"label": "a"})
    control = application.controller.context_control
    captured = control.read_context()
    bus = BaseEventBus()
    service = WritebackService(control, arb_names=lambda: ())
    guard = GuardService(application.state, access=ExperimentAccess())
    facet = WritebackControlFacet(
        state=application.state,
        guard=guard,
        writeback=service,
        resource_versions=application.state.version.snapshot,
        bus=bus,
    )
    application.remote.writeback_control = facet
    tab_id = application.controller.tab_control.new_tab("pointed")
    draft = service.create_draft(
        [ParameterWriteback("general.ext.flux_value", "Flux", proposed_value=2.0)],
        parameters=captured,
        experiment_name="test/point-admission",
    )
    commit_analysis(
        application.state,
        tab_id,
        AnalysisPaneState(
            result=object(),
            plots=None,
            params=application.state.get_tab(tab_id).analysis.params,
            writeback_draft=draft,
        ),
    )

    application.controller.setup_control.open_entry(EntryRequest("entry"))
    before = entry_resource(application).ledger.events()
    try:
        if frontend == "application":
            facet.apply_writeback_for_pane(tab_id, "analysis")
        else:
            reply = application.rpc(
                "tab.writeback_write",
                {
                    "tab_id": tab_id,
                    "subtab_id": "analysis",
                    "submission_id": application.result(
                        "tab.writeback_begin", {"tab_id": tab_id}
                    )["submission_id"],
                    "write": [{"id": "param-1"}],
                },
            )
            assert reply["ok"] is True, reply
        assert control.read_context().target == SavedEntry(entry_name="entry")
        assert control.read_context().value("general.ext.flux_value") == 2.0
        assert len(entry_resource(application).ledger.events()) == len(before) + 1
        application.result("point.new", {"label": "b", "mode": "empty"})
        facet.apply_writeback_for_pane(tab_id, "analysis")
        assert (
            not control.read_target(require_target(captured))
            .lookup("general.ext.flux_value")
            .present
        )
        selected = control.read_context()
        assert selected.value("general.ext.flux_value") == 2.0
        assert require_target(selected).point == "b"
        assert len(entry_resource(application).ledger.events()) == len(before) + 2
        application.controller.setup_control.new_entry(EntryRequest("other"))
        application.result("point.new", {"label": "c"})
        facet.apply_writeback_for_pane(tab_id, "analysis")
        assert control.read_context().value("general.ext.flux_value") == 2.0
        assert require_target(control.read_context()).point == "c"
        assert len(entry_resource(application).ledger.events()) == 1
    finally:
        service.teardown_draft(draft)
        application.state.get_tab(tab_id).analysis.writeback_draft = None


@pytest.mark.parametrize("frontend", ["application", "wire"])
def test_writeback_without_entry_rejects_before_appending_ledger(
    application: PointApplication,
    parameter_writeback_pane: tuple[str, ParameterContext],
    frontend: str,
) -> None:
    tab_id, _ = parameter_writeback_pane
    entry = entry_resource(application)
    before = entry.ledger.events()
    application.controller.session_services.context.clear_context()
    if frontend == "application":
        with pytest.raises(FailedPreconditionError) as caught:
            application.remote.writeback_control.apply_writeback_for_pane(
                tab_id, "analysis"
            )
        assert caught.value.reason_code == "no_context"
    else:
        reply = application.rpc(
            "tab.writeback_write",
            {
                "tab_id": tab_id,
                "subtab_id": "analysis",
                "submission_id": application.result(
                    "tab.writeback_begin", {"tab_id": tab_id}
                )["submission_id"],
                "write": [{"id": "param-1"}],
            },
        )
        assert reply["ok"] is False
        error = reply["error"]
        assert isinstance(error, dict)
        assert error["code"] == ErrorCode.PRECONDITION_FAILED
        assert error["reason"] == "no_context"
    assert entry.ledger.events() == before


@pytest.fixture
def parameter_writeback_pane(
    application: PointApplication,
) -> Iterator[tuple[str, ParameterContext]]:
    application.open_entry()
    application.result("point.new", {"label": "a"})
    control = application.controller.context_control
    captured = control.read_context()
    control.set_parameter(require_target(captured), "general.ext.a", 1.0)
    control.set_parameter(require_target(captured), "general.ext.b", 2.0)
    bus = BaseEventBus()
    service = WritebackService(control, arb_names=lambda: ())
    facet = WritebackControlFacet(
        state=application.state,
        guard=GuardService(application.state, access=ExperimentAccess()),
        writeback=service,
        resource_versions=application.state.version.snapshot,
        bus=bus,
    )
    application.remote.writeback_control = facet
    tab_id = application.controller.tab_control.new_tab("pointed")
    draft = service.create_draft(
        [
            ParameterWriteback("general.ext.a", "A", proposed_value=11.0),
            ParameterWriteback("general.ext.b", "B", proposed_value=22.0),
        ],
        parameters=captured,
        experiment_name="test/point-admission",
    )
    commit_analysis(
        application.state,
        tab_id,
        AnalysisPaneState(
            result=object(),
            plots=None,
            params=application.state.get_tab(tab_id).analysis.params,
            writeback_draft=draft,
        ),
    )
    registry = Registry()
    registry.register("pointed", PointAdapter)
    tab = TabService(
        application.state,
        registry,
        service,
        form_resources(application.state),
        lambda aliases: form_sources(application.state, aliases),
    )
    application.remote.tab_control = TabControlFacet(
        state=application.state,
        tab=tab,
        workspace=WorkspaceService(application.state, tab, bus),
        bus=bus,
        load_tab_result=application.controller.run_analyze_control.load_tab_result,
    )
    try:
        yield tab_id, captured
    finally:
        service.teardown_draft(draft)
        application.state.get_tab(tab_id).analysis.writeback_draft = None
