"""Writeback Edit dialogs: scalar parameter values and module item Forms."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtGui import QAction
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QPushButton,
)
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.pane_commit import commit_post_analysis
from zcu_tools.gui.app.measure.state import PostAnalysisPaneState, State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.writeback import ParameterWriteback, WritebackValues
from zcu_tools.gui.app.measure.writeback.widget import WritebackWidget
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.form_adapter import module_form
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.widgets.form import FormEditor

from tests._filesystem import DIRECTORY_OPEN_ERROR
from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._record_support import require_target
from tests.gui.app.measure._synthetic import (
    MODULE_PROPOSAL_TARGET,
    MODULE_PROPOSAL_WAVEFORM,
    ModuleProposalAdapter,
    SampleParams,
)
from tests.gui.widgets.form._text_input import commit_text_on_focus_loss as commit_text
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


def _preview(item: ParameterWriteback) -> dict[str, WritebackValues]:
    return {
        item.session_id: WritebackValues(
            actual_path=item.target_name,
            current=None,
            proposed=item.proposed_value,
            unit=None,
        )
    }


@pytest.fixture
def control_factory() -> Callable[[], MagicMock]:
    """Construct the controller collaborators used by local writeback dialogs."""
    return MagicMock


@pytest.fixture
def widget_factory(
    control_factory: Callable[[], MagicMock], qapp: QApplication
) -> Iterator[Callable[[ParameterWriteback, MagicMock | None], WritebackWidget]]:
    widgets: list[WritebackWidget] = []

    def create(item: ParameterWriteback, ctrl: MagicMock | None) -> WritebackWidget:
        if ctrl is None:
            ctrl = control_factory()
        ctrl.writeback_control.get_writeback_values_for_pane.side_effect = (
            lambda *args, **kwargs: _preview(item)
        )
        widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")
        widget.populate([item], _preview(item))
        widgets.append(widget)
        return widget

    yield create
    for widget in widgets:
        for dialog in widget.findChildren(QDialog):
            if dialog.isVisible():
                dialog.reject()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _edit_button(widget: WritebackWidget) -> QPushButton:
    return next(
        button for button in widget.findChildren(QPushButton) if button.text() == "Edit"
    )


def _find_open_dialog(title_prefix: str) -> QDialog:
    from qtpy.QtWidgets import QApplication

    # Match by title prefix: WA_DeleteOnClose defers destruction to the event
    # loop (which does not spin between tests), so a closed dialog from a prior
    # test can still linger in topLevelWidgets — picking the last one blindly
    # grabs the wrong dialog.
    dialogs = [
        w
        for w in QApplication.topLevelWidgets()
        if isinstance(w, QDialog)
        and w.isVisible()
        and w.windowTitle().startswith(title_prefix)
    ]
    assert dialogs, f"no open dialog titled {title_prefix!r}"
    return dialogs[-1]


def test_edit_md_item_can_change_target_name(qapp, widget_factory, control_factory):
    """The parameter Edit dialog exposes an editable apply-as (target_name)."""
    item = ParameterWriteback(
        target_name="r_f", description="freq", proposed_value=6000.0
    )
    item.session_id = "param-1"

    widget = widget_factory(item, None)
    _edit_button(widget).click()
    dialog = _find_open_dialog("Edit value —")
    try:
        name_edit = dialog.findChild(QLineEdit, "writebackApplyAs")
        value_edit = dialog.findChild(QLineEdit, "writebackValue")
        assert name_edit.text() == "r_f"

        name_edit.setText("r_f_tuned")
        value_edit.setText("6100.0")
        save_btn = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Save"
        )
        save_btn.click()

        assert item.target_name == "r_f_tuned"
        assert item.proposed_value == 6100.0
        # session_id (the stable id) is untouched by a retarget.
        assert item.session_id == "param-1"
    finally:
        dialog.close()


def test_edit_md_item_rejects_blank_target_name(qapp, widget_factory, control_factory):
    item = ParameterWriteback(
        target_name="r_f", description="freq", proposed_value=6000.0
    )
    item.session_id = "param-1"

    widget = widget_factory(item, None)
    _edit_button(widget).click()
    dialog = _find_open_dialog("Edit value —")
    try:
        name_edit = dialog.findChild(QLineEdit, "writebackApplyAs")
        name_edit.setText("   ")
        save_btn = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Save"
        )
        assert not save_btn.isEnabled()
        save_btn.click()

        # Blank name is rejected: target_name unchanged, dialog stays open.
        assert item.target_name == "r_f"
    finally:
        dialog.close()


def test_edit_md_item_parses_complex_value(qapp, widget_factory, control_factory):
    """A complex md value coerces the user's "re+imj" input to complex (not str)."""
    item = ParameterWriteback(
        target_name="g_center", description="|g> centre", proposed_value=complex(1, 2)
    )
    item.session_id = "param-1"

    widget = widget_factory(item, None)
    _edit_button(widget).click()
    dialog = _find_open_dialog("Edit value —")
    try:
        value_edit = dialog.findChild(QLineEdit, "writebackValue")
        value_edit.setText("3-4j")
        save_btn = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Save"
        )
        save_btn.click()

        assert item.proposed_value == complex(3, -4)
        assert isinstance(item.proposed_value, complex)
    finally:
        dialog.close()


def test_edit_md_item_rejects_unparseable_complex(
    qapp, widget_factory, control_factory
):
    """An unparseable complex input fast-fails and leaves the original value."""
    item = ParameterWriteback(
        target_name="g_center", description="|g> centre", proposed_value=complex(1, 2)
    )
    item.session_id = "param-1"

    widget = widget_factory(item, None)
    _edit_button(widget).click()
    dialog = _find_open_dialog("Edit value —")
    try:
        value_edit = dialog.findChild(QLineEdit, "writebackValue")
        value_edit.setText("not a number")
        save_btn = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Save"
        )
        assert not save_btn.isEnabled()
        save_btn.click()

        # Invalid input rejected: value unchanged, dialog stays open.
        assert item.proposed_value == complex(1, 2)
    finally:
        dialog.close()


class _WidgetController(Controller):
    """Publish synthetic busy state for the isolated Qt fixture."""

    def set_running(self, tab_id: str, *, running: bool) -> None:
        """Publish a synthetic busy state through the real State owner."""
        self._state.set_tab_running(tab_id, running=running)


@pytest.fixture(params=["analysis", "post_analysis"])
def module_widget(tmp_path: Path, qapp: QApplication, request: pytest.FixtureRequest):
    """Analyze a caller-registered adapter and view its real persistent draft."""
    before = registry_state()
    register_fakes()
    installed = registry_state()
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    registry = Registry()
    registry.register("synthetic/module", ModuleProposalAdapter)
    ctrl = _WidgetController(
        state,
        registry,
        None,
        BaseEventBus(),
        progress_transport=DirectProgressTransport(),
        project_root=str(tmp_path),
    )
    widget: WritebackWidget | None = None
    tab_id: str | None = None
    try:
        ctrl.setup_control.new_entry(EntryRequest("offline"))
        ctrl.setup_control.new_point(PointCreateRequest("a", "", CommandOrigin("gui")))
        context = ctrl.context_control
        context.add_component("Q1", "fake/drive/a", {})
        context.set_waveform(
            require_target(context.read_context()),
            MODULE_PROPOSAL_WAVEFORM,
            ConstWaveformCfg(length=0.1),
        )
        context.set_parameter(
            require_target(context.read_context()), "general.ext.gain", 0.4
        )
        board = white_noise()
        ctrl.session_services.soc_connection.install_soc(board.soc, board.soccfg)
        tab_id = ctrl.tab_control.new_tab("synthetic/module")
        ctrl.run_analyze_control.start_run(tab_id)
        # The admitted synthetic run returns; pump its terminal owner delivery.
        qt_until(
            lambda: ctrl.active_operation_count() == 0,
            label="module writeback run terminal delivery",
        )
        run = ctrl.tab_control.get_tab_snapshot(tab_id).run
        assert run is not None and run.result is not None
        ctrl.run_analyze_control.analyze(tab_id, SampleParams())
        # The admitted fit returns; its terminal owner delivery installs the draft.
        qt_until(
            lambda: ctrl.active_operation_count() == 0,
            label="module writeback analysis terminal delivery",
        )
        analysis = state.get_tab(tab_id).analysis
        assert analysis.result is not None and analysis.writeback_draft is not None
        pane: Literal["analysis", "post_analysis"] = request.param
        if pane == "post_analysis":
            commit_post_analysis(
                state,
                tab_id,
                PostAnalysisPaneState(
                    result=analysis.result,
                    plots=None,
                    params=state.get_tab(tab_id).post_analysis.params,
                    writeback_draft=analysis.writeback_draft,
                ),
            )
        widget = WritebackWidget(ctrl, tab_id=tab_id, pane=pane)
        widget.resize(700, 250)
        widget.show()
        yield ctrl, widget, tab_id, pane
        assert registry_state() == installed
    finally:
        ctrl.close()
        if widget is not None:
            for dialog in widget.findChildren(QDialog):
                dialog.reject()
            widget.deleteLater()
        if tab_id is not None and ctrl.tab_control.has_tab(tab_id):
            ctrl.tab_control.close_tab(tab_id)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        restore_registry(before)


def _populate_module(module_widget):
    ctrl, widget, tab_id, pane = module_widget
    snapshot = ctrl.tab_control.get_tab_snapshot(tab_id).analysis
    assert snapshot is not None
    widget.populate(
        snapshot.writeback_items,
        ctrl.writeback_control.get_writeback_values_for_pane(tab_id, pane),
    )
    return module_widget


def _pulse(ctrl: Controller, name: str) -> PulseSpec:
    spec = ctrl.context_control.read_context().library.module_spec(name)
    assert isinstance(spec, PulseSpec)
    return spec


def _module_dialog(widget: WritebackWidget) -> QDialog:
    _edit_button(widget).click()
    dialog = _find_open_dialog("Edit module —")
    assert dialog.findChild(FormEditor) is not None
    return dialog


def _input(dialog: QDialog, path: str) -> QLineEdit:
    control = dialog.findChild(QLineEdit, f"form.input.{path}")
    assert control is not None, f"missing public Form input {path}"
    return control


def _apply(widget: WritebackWidget) -> QPushButton:
    control = widget.findChild(QPushButton, "writebackApply")
    assert control is not None
    return control


@pytest.fixture
def module_ui(module_widget, tmp_path, request):
    _, widget, _, _ = _populate_module(module_widget)
    driver = ComponentDriver(widget)
    driver.resize_window(900, 400)
    evidence = Path(os.environ.get("WRITEBACK_UI_EVIDENCE", str(tmp_path)))
    evidence = evidence / request.node.name
    return driver, evidence


def test_module_cancel_then_save_updates_only_saved_ledger(module_ui):
    driver, evidence = module_ui
    original = driver.observe(object_name="writebackProposed")
    driver.screenshot(evidence / "01-ledger.png")
    driver.click("Edit")
    editor = ComponentDriver(_find_open_dialog("Edit module —"))
    editor.resize_window(700, 700)
    editor.screenshot(evidence / "02-edit.png")
    editor.enter_text(object_name="form.input.freq", value="5100.125")
    editor.screenshot(evidence / "03-changed.png")
    editor.click("Cancel")
    driver.screenshot(evidence / "04-cancelled.png")
    assert not driver.dialogs()
    assert driver.observe(object_name="writebackProposed").text == original.text
    assert driver.observe(object_name="writebackProposed").tooltip == original.tooltip
    driver.click("Edit")
    editor = ComponentDriver(_find_open_dialog("Edit module —"))
    editor.resize_window(700, 700)
    editor.screenshot(evidence / "05-reopen.png")
    assert editor.text_input(object_name="form.input.freq").text != "5100.125"
    editor.enter_text(object_name="form.input.freq", value="5100.125")
    editor.screenshot(evidence / "06-changed.png")
    editor.click("Save")
    driver.screenshot(evidence / "07-saved.png")
    assert not driver.dialogs()
    assert "5100.125" in driver.observe(object_name="writebackProposed").tooltip


def test_module_save_keeps_captured_expression_and_waveform_key(module_widget, module_ui):
    ctrl, widget, tab_id, pane = module_widget
    driver, evidence = module_ui
    driver.click("Edit")
    editor = ComponentDriver(_find_open_dialog("Edit module —"))
    editor.resize_window(700, 700)
    editor.screenshot(evidence / "01-edit.png")
    editor.enter_text(object_name="form.input.freq", value="5100.125")
    editor.screenshot(evidence / "02-frequency.png")
    editor.invoke_context_action(
        object_name="form.input.gain", action="Expression",
        evidence=evidence / "03-expression-menu.png",
    )
    editor.screenshot(evidence / "04-expression.png")
    editor.enter_text(object_name="form.input.gain", value="general.ext.gain / 2")
    editor.screenshot(evidence / "05-gain.png")
    editor.enter_text(object_name="writebackApplyAs", value="Q1.pulses.retuned")
    editor.screenshot(evidence / "06-name.png")
    context = ctrl.context_control
    context.set_parameter(require_target(context.read_context()), "general.ext.gain", 0.8)
    editor.click("Save")
    driver.screenshot(evidence / "07-saved.png")
    form = ctrl.writeback_control.get_writeback_item_form_for_pane(tab_id, pane, "ml-1")
    spec = form.spec()
    assert isinstance(spec, PulseSpec)
    assert spec.freq == 5100.125
    assert spec.gain == pytest.approx(0.2)
    assert spec.waveform == MODULE_PROPOSAL_WAVEFORM
    widget.apply_requested.connect(
        lambda: ctrl.writeback_control.apply_writeback_for_pane(tab_id, pane)
    )
    driver.click("Apply Selected")
    driver.screenshot(evidence / "08-applied.png")
    assert _pulse(ctrl, "Q1.pulses.retuned") == spec


@pytest.mark.parametrize("busy", [False, True], ids=["convert", "busy-save-refusal"])
def test_module_option_command_is_local_until_save(module_widget, module_ui, busy):
    ctrl, _, tab_id, pane = module_widget
    driver, evidence = module_ui
    driver.click("Edit")
    editor = ComponentDriver(_find_open_dialog("Edit module —"))
    editor.resize_window(700, 700)
    editor.screenshot(evidence / "01-edit.png")
    form = ctrl.writeback_control.get_writeback_item_form_for_pane(tab_id, pane, "ml-1")
    before = form.state.input.data
    captured = form.parameters.library.waveform_spec(MODULE_PROPOSAL_WAVEFORM)
    context = ctrl.context_control
    context.set_waveform(
        require_target(context.read_context()), MODULE_PROPOSAL_WAVEFORM,
        ConstWaveformCfg(length=0.75),
    )
    ctrl.set_running(tab_id, running=busy)
    try:
        editor.invoke_context_action(
            object_name="form.input.waveform", action="Convert to inline",
            evidence=evidence / "02-convert-menu.png",
        )
        editor.screenshot(evidence / "03-converted.png")
        assert form.state.input.data == before
        editor.click("Save")
        driver.screenshot(evidence / "04-save.png")
        if busy:
            assert driver.dialogs()
            assert "busy" in editor.observe(object_name="writebackEditError").text.lower()
            assert form.state.input.data == before
        else:
            assert not driver.dialogs()
            assert form.spec().waveform == captured
    finally:
        ctrl.set_running(tab_id, running=False)


@pytest.mark.parametrize("invalid_kind", ["unparsed", "expression"])
def test_invalid_module_copy_keeps_ledger_and_can_be_cancelled(
    module_ui, invalid_kind
):
    driver, evidence = module_ui
    original = driver.observe(object_name="writebackProposed")
    driver.click("Edit")
    editor = ComponentDriver(_find_open_dialog("Edit module —"))
    editor.resize_window(700, 700)
    editor.screenshot(evidence / "01-edit.png")
    if invalid_kind == "expression":
        editor.invoke_context_action(
            object_name="form.input.gain", action="Expression",
            evidence=evidence / "02-expression-menu.png",
        )
        editor.screenshot(evidence / "03-expression.png")
    editor.enter_text(object_name="form.input.gain", value="missing.value")
    editor.screenshot(evidence / "04-invalid.png")
    assert not editor.observe("Save").enabled
    assert "gain" in editor.observe(object_name="writebackEditError").text
    editor.click("Cancel")
    driver.screenshot(evidence / "05-cancelled.png")
    assert not driver.dialogs()
    assert driver.observe(object_name="writebackProposed").text == original.text
    assert driver.observe(object_name="writebackProposed").tooltip == original.tooltip
    assert driver.observe("Apply Selected").enabled


def test_module_dialog_reports_retired_draft_without_opening_form(module_widget, qapp):
    ctrl, widget, tab_id, _pane = _populate_module(module_widget)
    ctrl.tab_control.close_tab(tab_id)
    _edit_button(widget).click()
    dialog = _find_open_dialog("Edit module —")
    assert tab_id in "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert dialog.findChild(FormEditor) is None
    dialog.accept()


def test_busy_edit_request_shows_guard_reason_without_opening_dialog(
    module_widget, qapp
):
    ctrl, widget, tab_id, _pane = _populate_module(module_widget)
    ctrl.set_running(tab_id, running=True)
    try:
        _edit_button(widget).click()
        dialog = _find_open_dialog("Edit module —")
        assert (
            "busy"
            in "\n".join(label.text() for label in dialog.findChildren(QLabel)).lower()
        )
        assert dialog.findChild(FormEditor) is None
        dialog.accept()
        assert (
            MODULE_PROPOSAL_TARGET
            not in ctrl.context_control.read_context().library.module_keys()
        )
    finally:
        ctrl.set_running(tab_id, running=False)


@pytest.fixture
def module_host(module_widget, qapp):
    """Mount the same registered caller and real EventBus in its Qt host."""
    ctrl, _isolated, tab_id, pane = module_widget
    presenter = RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=presenter)
    window.add_tab_widget(tab_id, "synthetic/module")
    window.refresh_tab_writeback(tab_id)
    window.select_tab_pane(tab_id, "analysis")
    widget = window.findChild(WritebackWidget)
    assert widget is not None
    window.show()
    try:
        yield ctrl, window, widget, tab_id, pane, presenter
    finally:
        # Publish the real tab close while its subscribed Qt host is still live.
        if ctrl.tab_control.has_tab(tab_id):
            ctrl.tab_control.close_tab(tab_id)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


@pytest.mark.parametrize("module_widget", ["analysis"], indirect=True)
def test_tab_retirement_closes_its_open_writeback_editor(module_host, tmp_path):
    ctrl, window, _widget, tab_id, _pane, _presenter = module_host
    driver = ComponentDriver(window)
    driver.click("Edit")
    dialog = _find_open_dialog("Edit module —")
    editor = ComponentDriver(dialog)
    editor.screenshot(tmp_path / "01-open-writeback.png")
    ctrl.tab_control.close_tab(tab_id)
    assert not editor.visible().widgets
    driver.screenshot(tmp_path / "02-retired-tab.png")


@pytest.mark.parametrize("module_widget", ["analysis"], indirect=True)
def test_host_apply_presents_expected_busy_error_without_losing_draft(module_host):
    ctrl, window, widget, tab_id, pane, presenter = module_host
    dialog = _module_dialog(widget)
    commit_text(_input(dialog, "freq"), "5100.125")
    form = ctrl.writeback_control.get_writeback_item_form_for_pane(tab_id, pane, "ml-1")
    before = form.state.input.data
    ctrl.set_running(tab_id, running=True)
    try:
        try:
            window.apply_tab_writeback(tab_id, pane=pane)
        except ExpectedError as error:
            pytest.fail(
                f"Host must present the writeback refusal instead of raising: {error}"
            )
        assert len(presenter.calls) == 1 and presenter.calls[0].kind == "critical"
        assert "busy" in presenter.calls[0].message.lower()
        assert (
            MODULE_PROPOSAL_TARGET
            not in ctrl.context_control.read_context().library.module_keys()
        )
        assert form.state.input.data == before
        assert dialog.isVisible() and _input(dialog, "freq").text() == "5100.125"
    finally:
        ctrl.set_running(tab_id, running=False)


@pytest.mark.parametrize("module_widget", ["analysis"], indirect=True)
def test_controller_apply_propagates_unexpected_storage_error(module_host):
    ctrl, _window, _widget, tab_id, pane, presenter = module_host
    context = ctrl.context_control
    storage = context.capture_storage(context.read_context())
    ledger = storage.result_path / "records" / "ledger.jsonl"
    backup = ledger.with_suffix(".saved")
    if ledger.exists():
        ledger.rename(backup)
    ledger.mkdir()
    try:
        # The application propagates storage failures; the Qt host presents them.
        with pytest.raises(DIRECTORY_OPEN_ERROR):
            ctrl.writeback_control.apply_writeback_for_pane(tab_id, pane)
        assert presenter.calls == []
        assert (
            MODULE_PROPOSAL_TARGET not in context.read_context().library.module_keys()
        )
    finally:
        ledger.rmdir()
        if backup.exists():
            backup.rename(ledger)
