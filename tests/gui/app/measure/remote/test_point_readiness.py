"""Entry selection admission at real application, view and wire seams."""

from typing import NoReturn

import pytest
from qtpy.QtWidgets import QLabel, QPushButton
from zcu_tools.gui.app.measure.adapter import ParameterWriteback
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.writeback import WritebackService
from zcu_tools.gui.app.measure.services.writeback_control import WritebackControlFacet
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.remote.errors import ErrorCode
from zcu_tools.gui.session.context_types import EntryRequest, SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.resources.entry import ResultEntry

from tests.gui._dialog_fakes import RecordingDialogPresenter

from ._point_application import PointApplication


def entry_resource(application: PointApplication) -> ResultEntry:
    info = application.controller.context_control.get_entry_info()
    assert info is not None
    return ResultEntry.open(
        info.name,
        result_root=info.result_path.parent,
        database_root=info.database_path.parent,
    )


def unused_resolution(_parameters: ParameterContext | None) -> NoReturn:
    raise AssertionError("Writeback admission must not assemble Run cfg")


@pytest.mark.parametrize("command", ["new", "open"])
def test_entry_commands_require_explicit_point_before_application_and_wire_reads(
    application: PointApplication, command: str
) -> None:
    if command == "open":
        application.open_entry()
        application.result("point.new", {"label": "existing"})
    application.result(f"entry.{command}", {"name": "entry"})
    assert application.state.session_env.parameters is None
    assert application.state.session_env.active_label == ""
    assert application.controller.context_control.get_entry_info() is not None
    labels = application.result("point.list", {})["points"]
    assert labels == (
        [] if command == "new" else [{"label": "existing", "description": ""}]
    )
    control = application.controller.context_control
    with pytest.raises(FailedPreconditionError) as caught:
        control.read_context()
    assert caught.value.reason_code == "no_context"
    with pytest.raises(FailedPreconditionError) as caught:
        control.add_component("Q1", "fake/drive/a", {})
    assert caught.value.reason_code == "no_context"
    error = application.rpc("context.snapshot", {})["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "no_context"
    tab_id = application.controller.new_tab("pointed")
    with pytest.raises(FailedPreconditionError) as caught:
        application.controller.start_run(
            tab_id, application.controller.cfg_resources.lookup(tab_id).observe().ref
        )
    assert caught.value.reason_code == "no_active_context"
    ref = application.controller.cfg_resources.lookup(tab_id).observe().ref
    reply = application.rpc(
        "tab.run_start",
        {
            "tab_id": tab_id,
            "expected": {"cfg_id": ref.cfg_id, "revision": str(ref.revision)},
        },
    )
    assert reply["ok"] is False
    assert isinstance(reply["error"], dict)
    assert reply["error"]["code"] == ErrorCode.PRECONDITION_FAILED
    application.result(
        "point.new" if command == "new" else "point.use",
        {"label": "selected" if command == "new" else "existing"},
    )
    capture = control.read_context()
    control.set_parameter(capture.target, "general.ext.flux_value", 2.0)
    assert control.read_target(capture.target).value("general.ext.flux_value") == 2.0


def test_unselected_point_disables_run_writeback_and_inspect_with_next_step(
    application: PointApplication, qapp
) -> None:
    application.open_entry()
    tab_id = application.controller.new_tab("pointed")
    application.state.update_tab_analyze(tab_id, object(), None)
    window = MainWindow(
        application.controller, dialog_presenter=RecordingDialogPresenter()
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
        assert not tab.writeback_widget.isEnabled()
        assert tab.writeback_widget.toolTip().strip()
        error = editor.findChild(QLabel, "component_error")
        assert error is not None
        assert not error.isHidden()
        assert error.text().strip()
        add = next(
            button
            for button in editor.findChildren(QPushButton)
            if button.text() == "Add"
        )
        set_field = next(
            button
            for button in editor.findChildren(QPushButton)
            if button.text() == "Set field"
        )
        assert not add.isEnabled() and not set_field.isEnabled()
        assert not editor.edit_parameter("general.ext.flux_value", 1.0)
        application.result("point.new", {"label": "selected"})
        editor.refresh()
        assert add.isEnabled()
        assert editor.edit_parameter("general.ext.flux_value", 1.0)
    finally:
        editor.deleteLater()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("frontend", ["application", "wire"])
def test_writeback_requires_selection_after_reopen_but_keeps_captured_destination(
    application: PointApplication, frontend: str
) -> None:
    application.open_entry()
    application.result("point.new", {"label": "a"})
    control = application.controller.context_control
    captured = control.read_context()
    bus = BaseEventBus()
    editor = CfgEditorService(
        application.controller,
        application.controller,
        application.controller,
        application.controller,
        bus,
    )

    service = WritebackService(editor, write_port=control)
    guard = GuardService(
        application.state,
        captured_parameters=lambda aliases: control.read_context(aliases=aliases),
        cfg_resolution=unused_resolution,
    )
    facet = WritebackControlFacet(
        state=application.state,
        guard=guard,
        writeback=service,
        resource_versions=application.state.version.snapshot,
        bus=bus,
    )
    application.remote.writeback_control = facet
    tab_id = application.controller.new_tab("pointed")
    draft = service.create_draft(
        [ParameterWriteback("general.ext.flux_value", "Flux", proposed_value=2.0)],
        parameters=captured,
        experiment_name="test/point-admission",
    )
    application.state.update_tab_analyze(tab_id, object(), None, writeback_draft=draft)
    application.controller.setup_control.open_entry(EntryRequest("entry"))
    before = entry_resource(application).ledger.events()
    try:
        if frontend == "application":
            with pytest.raises(FailedPreconditionError) as caught:
                facet.apply_writeback_for_pane(tab_id, "analysis")
            assert caught.value.reason_code == "no_context"
        else:
            reply = application.rpc(
                "tab.writeback_apply", {"tab_id": tab_id, "subtab_id": "analysis"}
            )
            assert reply["ok"] is False
            error = reply["error"]
            assert isinstance(error, dict)
            assert error["code"] == ErrorCode.PRECONDITION_FAILED
        assert entry_resource(application).ledger.events() == before
        application.result("point.new", {"label": "b"})
        facet.apply_writeback_for_pane(tab_id, "analysis")
        assert (
            control.read_target(captured.target).value("general.ext.flux_value") == 2.0
        )
        assert not control.read_context().lookup("general.ext.flux_value").present
        assert control.read_context().target.point == "b"
        application.controller.setup_control.new_entry(EntryRequest("other"))
        application.result("point.new", {"label": "c"})
        with pytest.raises(FailedPreconditionError) as caught:
            facet.apply_writeback_for_pane(tab_id, "analysis")
        assert caught.value.reason_code == "entry_not_open"
    finally:
        service.teardown_draft(draft)
        application.state.get_tab(tab_id).analysis.writeback_draft = None
