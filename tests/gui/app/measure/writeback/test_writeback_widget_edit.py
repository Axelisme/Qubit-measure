"""Writeback Edit dialogs: scalar parameter values and module item Forms."""

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType
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
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
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
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._synthetic import (
    MODULE_PROPOSAL_TARGET,
    MODULE_PROPOSAL_WAVEFORM,
    ModuleProposalAdapter,
    SampleParams,
)
from tests.gui.app.measure.ui._qt_interactions import wait_until
from tests.gui.widgets.form._field_errors import field_error
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
            context.read_context().target,
            MODULE_PROPOSAL_WAVEFORM,
            ConstWaveformCfg(length=0.1),
        )
        context.set_parameter(context.read_context().target, "general.ext.gain", 0.4)
        tab_id = ctrl.new_tab("synthetic/module")
        ctrl.start_run(tab_id)
        wait_until(qapp, lambda: ctrl.active_operation_count() == 0)
        run = ctrl.get_tab_snapshot(tab_id).run
        assert run is not None and run.result is not None
        ctrl.analyze(tab_id, SampleParams())
        wait_until(qapp, lambda: ctrl.active_operation_count() == 0)
        analysis = state.get_tab(tab_id).analysis
        assert analysis.result is not None and analysis.writeback_draft is not None
        pane: Literal["analysis", "post_analysis"] = request.param
        if pane == "post_analysis":
            state.update_tab_post_analyze(
                tab_id,
                analysis.result,
                None,
                writeback_draft=analysis.writeback_draft,
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
        if tab_id is not None and ctrl.has_tab(tab_id):
            ctrl.tab_control.close_tab(tab_id)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        restore_registry(before)


def _populate_module(module_widget):
    ctrl, widget, tab_id, pane = module_widget
    snapshot = ctrl.get_tab_snapshot(tab_id).analysis
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


def test_module_dialog_edits_shared_form_and_applies_as_new_name(module_widget):
    ctrl, widget, tab_id, pane = _populate_module(module_widget)
    dialog = _module_dialog(widget)
    # This public owner returns the actual module Edit draft and fixed capture.
    form = ctrl.get_writeback_item_form_for_pane(tab_id, pane, "ml-1")
    assert MODULE_PROPOSAL_WAVEFORM in form.parameters.library.waveform_keys()
    current = form.state.input.data["waveform"]
    assert current == {"kind": "ref", "name": MODULE_PROPOSAL_WAVEFORM}
    options = module_form(PulseSpec).model.options(form.env)[("waveform",)]
    selected = next(option for option in options if option.matches(current))
    assert selected.label == MODULE_PROPOSAL_WAVEFORM
    assert selected.value == current
    waveform = dialog.findChild(QComboBox, "form.input.waveform")
    assert waveform is not None
    assert waveform.currentText() == selected.label
    assert waveform.currentData() == selected.value
    assert (selected.label, selected.value) in [
        (waveform.itemText(index), waveform.itemData(index))
        for index in range(waveform.count())
    ]
    assert not any(
        "(missing)" in waveform.itemText(index) for index in range(waveform.count())
    )
    assert not isinstance(form.state.result(form.env), FormErrors)
    initial = form.spec()
    assert isinstance(initial, PulseSpec)
    assert initial.waveform == MODULE_PROPOSAL_WAVEFORM
    commit_text(_input(dialog, "freq"), "5100.125")
    mode = dialog.findChild(QAction, "form.mode.gain")
    assert mode is not None
    mode.trigger()
    commit_text(_input(dialog, "gain"), "general.ext.gain / 2")
    name = dialog.findChild(QLineEdit, "writebackApplyAs")
    name.setText("Q1.pulses.retuned")
    name.editingFinished.emit()
    # A blank Apply-as does not erase the last destination.
    name.setText("  ")
    name.editingFinished.emit()
    assert name.text() == "Q1.pulses.retuned"
    assert widget.findChildren(QCheckBox)[0].text() == "Q1.pulses.retuned*"
    proposed = ctrl.writeback_control.get_writeback_values_for_pane(tab_id, pane)[
        "ml-1"
    ].proposed
    assert proposed["freq"] == 5100.125
    assert proposed["gain"] == pytest.approx(0.2)
    assert selected.matches(form.state.input.data["waveform"])
    assert not isinstance(form.state.result(form.env), FormErrors)
    assert proposed["waveform"] == MODULE_PROPOSAL_WAVEFORM
    # Preview and Apply use the opening capture, not subsequently changed values.
    context = ctrl.context_control
    context.set_parameter(context.read_context().target, "general.ext.gain", 0.8)
    widget.apply_requested.connect(lambda: ctrl.apply_writeback_for_pane(tab_id, pane))
    dialog.accept()
    _apply(widget).click()
    spec = _pulse(ctrl, "Q1.pulses.retuned")
    assert spec.freq == 5100.125
    assert spec.gain == pytest.approx(0.2)
    assert spec.waveform == MODULE_PROPOSAL_WAVEFORM
    assert MODULE_PROPOSAL_TARGET not in context.read_context().library.module_keys()


@pytest.mark.parametrize("busy", [False, True], ids=["convert", "busy-refusal"])
def test_module_option_command_uses_captured_source_and_current_admission(
    module_widget, busy: bool
):
    ctrl, widget, tab_id, pane = _populate_module(module_widget)
    errors: list[BaseException] = []

    def record_qt_exception(
        _kind: type[BaseException],
        error: BaseException,
        _traceback: TracebackType | None,
    ) -> None:
        errors.append(error)

    previous_hook = sys.excepthook
    try:
        sys.excepthook = record_qt_exception
        _edit_button(widget).click()
        assert not errors, f"Module Edit must open through the public Qt seam: {errors}"
        dialog = _find_open_dialog("Edit module —")
        form = ctrl.get_writeback_item_form_for_pane(tab_id, pane, "ml-1")
        captured = form.parameters.library.waveform_spec(MODULE_PROPOSAL_WAVEFORM)
        ctrl.apply_writeback_for_pane(tab_id, pane)
        assert ctrl.writeback_control.get_writeback_applied_for_pane(tab_id, pane)[
            "ml-1"
        ]
        before = form.state.input.data
        context = ctrl.context_control
        context.set_waveform(
            context.read_context().target,
            MODULE_PROPOSAL_WAVEFORM,
            ConstWaveformCfg(length=0.75),
        )
        action = dialog.findChild(QAction, "form.command.waveform.to_inline")
        assert action is not None
        if busy:
            ctrl.set_running(tab_id, running=True)
        action.trigger()
        assert not errors, f"Option commands must use the public editor route: {errors}"
        if busy:
            assert form.state.input.data == before
            assert "busy" in field_error(dialog, "waveform").lower()
            assert ctrl.writeback_control.get_writeback_applied_for_pane(tab_id, pane)[
                "ml-1"
            ]
        else:
            assert form.state.input.data["waveform"] != before["waveform"]
            result = form.spec()
            assert isinstance(result, PulseSpec)
            assert result.waveform == captured
            preview = ctrl.writeback_control.get_writeback_values_for_pane(tab_id, pane)
            assert preview["ml-1"].proposed["waveform"] == captured.model_dump(
                mode="python"
            )
            assert not ctrl.writeback_control.get_writeback_applied_for_pane(
                tab_id, pane
            )["ml-1"]
            ctrl.apply_writeback_for_pane(tab_id, pane)
            assert _pulse(ctrl, MODULE_PROPOSAL_TARGET).waveform == captured
        assert dialog.isVisible()
    finally:
        sys.excepthook = previous_hook
        ctrl.set_running(tab_id, running=False)


@pytest.mark.parametrize("invalid_kind", ["unparsed", "expression"])
def test_invalid_module_dialog_marks_row_refuses_apply_and_repairs(
    module_widget, invalid_kind: str
):
    ctrl, widget, tab_id, pane = _populate_module(module_widget)
    dialog = _module_dialog(widget)
    gain = _input(dialog, "gain")
    if invalid_kind == "expression":
        mode = dialog.findChild(QAction, "form.mode.gain")
        assert mode is not None
        mode.trigger()
        commit_text(_input(dialog, "gain"), "missing.value")
    else:
        commit_text(gain, "not-a-number")
    proposed = widget.findChild(QLabel, "writebackProposed")
    assert proposed is not None
    assert proposed.text() == "Invalid configuration"
    assert not _apply(widget).isEnabled()
    writes: list[bool] = []
    widget.apply_requested.connect(lambda: writes.append(True))
    _apply(widget).click()
    assert writes == []
    assert (
        MODULE_PROPOSAL_TARGET
        not in ctrl.context_control.read_context().library.module_keys()
    )
    if invalid_kind == "expression":
        with pytest.raises(LibraryEntryInvalid, match="gain"):
            ctrl.apply_writeback_for_pane(tab_id, pane)
    dialog.accept()
    # Closing drops only unparsed view text; invalid State input survives.
    assert _apply(widget).isEnabled() == (invalid_kind == "unparsed")
    assert (
        MODULE_PROPOSAL_TARGET
        not in ctrl.context_control.read_context().library.module_keys()
    )
    dialog = _module_dialog(widget)
    if invalid_kind == "expression":
        assert _input(dialog, "gain").text() == "missing.value"
    else:
        assert float(_input(dialog, "gain").text()) == pytest.approx(0.3)
    commit_text(_input(dialog, "gain"), "0.45")
    assert proposed.text().endswith(" changes")
    assert _apply(widget).isEnabled()
    widget.apply_requested.connect(lambda: ctrl.apply_writeback_for_pane(tab_id, pane))
    _apply(widget).click()
    assert _pulse(ctrl, MODULE_PROPOSAL_TARGET).gain == 0.45


def test_module_dialog_reports_retired_draft_without_opening_form(module_widget, qapp):
    ctrl, widget, tab_id, _pane = _populate_module(module_widget)
    ctrl.tab_control.close_tab(tab_id)
    _edit_button(widget).click()
    dialog = _find_open_dialog("Edit module —")
    assert tab_id in "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert dialog.findChild(FormEditor) is None
    dialog.accept()


@pytest.mark.parametrize("scenario", ["open-local", "closed-state"])
def test_module_validity_refresh_uses_current_view_or_repaired_state(
    module_widget, scenario: str
):
    ctrl, widget, tab_id, pane = _populate_module(module_widget)
    dialog = _module_dialog(widget)
    if scenario == "open-local":
        commit_text(_input(dialog, "gain"), "not-a-number")
        _populate_module(module_widget)
        proposed = widget.findChild(QLabel, "writebackProposed")
        # Old rows awaiting Qt deletion are not the new rendered preview.
        assert proposed is not None
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        proposed = widget.findChild(QLabel, "writebackProposed")
        assert proposed is not None and proposed.text() == "Invalid configuration"
        assert not _apply(widget).isEnabled()
    else:
        mode = dialog.findChild(QAction, "form.mode.gain")
        assert mode is not None
        mode.trigger()
        commit_text(_input(dialog, "gain"), "missing.value")
        dialog.accept()
        assert not _apply(widget).isEnabled()
        ctrl.set_writeback_item_for_pane(tab_id, pane, "ml-1", edits={("gain",): 0.5})
        _populate_module(module_widget)
        assert _apply(widget).isEnabled()


def test_busy_module_field_edit_is_rejected_and_dialog_keeps_authored_input(
    module_widget,
):
    ctrl, widget, tab_id, pane = _populate_module(module_widget)
    dialog = _module_dialog(widget)
    commit_text(_input(dialog, "freq"), "5100.125")
    form = ctrl.get_writeback_item_form_for_pane(tab_id, pane, "ml-1")
    before = form.state.input.data
    ctrl.set_running(tab_id, running=True)
    try:
        commit_text(_input(dialog, "gain"), "0.6")
        assert "busy" in field_error(dialog, "gain").lower()
        assert form.state.input.data == before
        assert dialog.isVisible()
        assert _input(dialog, "freq").text() == "5100.125"
        assert _input(dialog, "gain").text() == "0.6"
        assert (
            MODULE_PROPOSAL_TARGET
            not in ctrl.context_control.read_context().library.module_keys()
        )
    finally:
        ctrl.set_running(tab_id, running=False)


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
        if ctrl.has_tab(tab_id):
            ctrl.tab_control.close_tab(tab_id)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


@pytest.mark.parametrize("module_widget", ["analysis"], indirect=True)
def test_host_module_invalid_edit_keeps_row_dialog_and_draft(module_host):
    ctrl, window, widget, tab_id, pane, presenter = module_host
    dialog = _module_dialog(widget)
    commit_text(_input(dialog, "freq"), "5100.125")
    mode = dialog.findChild(QAction, "form.mode.gain")
    assert mode is not None
    mode.trigger()
    commit_text(_input(dialog, "gain"), "missing.value")
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    proposed = widget.findChild(QLabel, "writebackProposed")
    assert proposed is not None and proposed.text() == "Invalid configuration"
    assert dialog.isVisible() and _input(dialog, "freq").text() == "5100.125"
    assert not _apply(widget).isEnabled()
    with pytest.raises(LibraryEntryInvalid):
        ctrl.apply_writeback_for_pane(tab_id, pane)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    proposed = widget.findChild(QLabel, "writebackProposed")
    assert proposed is not None and proposed.text() == "Invalid configuration"
    selection = widget.findChild(QCheckBox)
    assert selection is not None and selection.isChecked()
    assert not _apply(widget).isEnabled()
    assert dialog.isVisible()
    assert _input(dialog, "freq").text() == "5100.125"
    assert _input(dialog, "gain").text() == "missing.value"
    assert (
        MODULE_PROPOSAL_TARGET
        not in ctrl.context_control.read_context().library.module_keys()
    )
    assert presenter.calls == []
    commit_text(_input(dialog, "gain"), "0.45")
    dialog.accept()
    window.apply_tab_writeback(tab_id, pane=pane)
    assert _pulse(ctrl, MODULE_PROPOSAL_TARGET).gain == 0.45
    assert _pulse(ctrl, MODULE_PROPOSAL_TARGET).freq == 5100.125


@pytest.mark.parametrize("module_widget", ["analysis"], indirect=True)
def test_host_apply_presents_expected_busy_error_without_losing_draft(module_host):
    ctrl, window, widget, tab_id, pane, presenter = module_host
    dialog = _module_dialog(widget)
    commit_text(_input(dialog, "freq"), "5100.125")
    form = ctrl.get_writeback_item_form_for_pane(tab_id, pane, "ml-1")
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
def test_host_apply_does_not_swallow_unexpected_storage_error(module_host):
    ctrl, window, _widget, tab_id, pane, presenter = module_host
    context = ctrl.context_control
    storage = context.capture_storage(context.read_context())
    ledger = storage.result_path / "records" / "ledger.jsonl"
    backup = ledger.with_suffix(".saved")
    if ledger.exists():
        ledger.rename(backup)
    ledger.mkdir()
    try:
        with pytest.raises(IsADirectoryError):
            window.apply_tab_writeback(tab_id, pane=pane)
        assert presenter.calls == []
        assert (
            MODULE_PROPOSAL_TARGET not in context.read_context().library.module_keys()
        )
    finally:
        ledger.rmdir()
        if backup.exists():
            backup.rename(ledger)
