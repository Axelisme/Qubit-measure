from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import pytest
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QCheckBox, QDialog, QLineEdit, QPushButton
from zcu_tools.gui.app.measure.adapter import ModuleWriteback, ParameterWriteback
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.services.writeback import WritebackValues
from zcu_tools.gui.app.measure.ui.writeback_widget import WritebackWidget
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.control_identity import ControlIdentity


def _preview(item: ParameterWriteback | ModuleWriteback) -> dict[str, WritebackValues]:
    return {
        item.session_id: WritebackValues(
            actual_path=item.target_name,
            current=None,
            proposed=item.proposed_value
            if isinstance(item, ParameterWriteback)
            else {},
            unit=None,
        )
    }


@pytest.fixture
def control_factory() -> Iterator[Callable[[], MagicMock]]:
    """Own each fake application authority until dependent views are deleted."""
    controllers: list[MagicMock] = []

    def create() -> MagicMock:
        ctrl = MagicMock(control_identity=ControlIdentity())
        controllers.append(ctrl)
        return ctrl

    yield create
    controllers.clear()


@pytest.fixture
def widget_factory(
    control_factory: Callable[[], MagicMock], qapp: QApplication
) -> Iterator[
    Callable[[ParameterWriteback | ModuleWriteback, MagicMock | None], WritebackWidget]
]:
    widgets: list[WritebackWidget] = []

    def create(
        item: ParameterWriteback | ModuleWriteback, ctrl: MagicMock | None
    ) -> WritebackWidget:
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
    dialog = _find_open_dialog("Edit Value:")
    try:
        line_edits = dialog.findChildren(QLineEdit)
        # First field is "Apply as" (target_name), second is "Value".
        name_edit, value_edit = line_edits[0], line_edits[1]
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
    dialog = _find_open_dialog("Edit Value:")
    try:
        name_edit = dialog.findChildren(QLineEdit)[0]
        name_edit.setText("   ")
        save_btn = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Save"
        )
        # The validation error pops a modal QMessageBox; stub it so the test
        # does not block on it.
        with patch(
            "zcu_tools.gui.app.measure.ui.writeback_widget.QMessageBox.critical"
        ):
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
    dialog = _find_open_dialog("Edit Value:")
    try:
        value_edit = dialog.findChildren(QLineEdit)[1]
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
    dialog = _find_open_dialog("Edit Value:")
    try:
        value_edit = dialog.findChildren(QLineEdit)[1]
        value_edit.setText("not a number")
        save_btn = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Save"
        )
        with patch(
            "zcu_tools.gui.app.measure.ui.writeback_widget.QMessageBox.critical"
        ):
            save_btn.click()

        # Invalid input rejected: value unchanged, dialog stays open.
        assert item.proposed_value == complex(1, 2)
    finally:
        dialog.close()


def test_edit_cfg_item_reports_draft_resolution_failure(
    qapp, widget_factory, control_factory
):
    item = ModuleWriteback(
        target_name="readout_rf", description="A module", edit_schema=MagicMock()
    )
    item.session_id = "ml-1"
    ctrl = control_factory()
    ctrl.get_writeback_item_draft_for_pane.side_effect = RuntimeError("draft closed")
    widget = widget_factory(item, ctrl)

    with patch(
        "zcu_tools.gui.app.measure.ui.writeback_widget.QMessageBox.critical"
    ) as critical:
        _edit_button(widget).click()

    critical.assert_called_once_with(widget, "Unable to edit writeback", "draft closed")


def test_edit_cfg_item_can_change_target_name(qapp, widget_factory, control_factory):
    """The module/waveform Edit dialog exposes an editable apply-as that commits
    on editingFinished."""
    item = ModuleWriteback(
        target_name="readout_rf", description="A module", edit_schema=MagicMock()
    )
    item.session_id = "ml-1"

    ctrl = control_factory()
    draft = MeasureCfgBindings(ctrl).new_draft(
        CfgSchema(
            spec=CfgSectionSpec(fields={}),
            value=CfgSectionValue(fields={}),
        ),
        parameters=None,
    )
    ctrl.get_writeback_item_draft_for_pane.return_value = draft
    ctrl.get_writeback_applied_for_pane.return_value = {"ml-1": True}
    widget = widget_factory(item, ctrl)
    cb = widget.findChildren(QCheckBox)[0]
    _edit_button(widget).click()
    dialog = _find_open_dialog("Edit Config:")
    try:
        name_edit = dialog.findChildren(QLineEdit)[0]
        assert name_edit.text() == "readout_rf"

        name_edit.setText("readout_rf_tuned")
        name_edit.editingFinished.emit()
        assert item.target_name == "readout_rf_tuned"

        ctrl.get_writeback_applied_for_pane.return_value = {"ml-1": False}
        draft.on_change.emit()
        assert cb.text() == "readout_rf_tuned*"
        assert cb.font().bold()

        # Blank reverts to the previous name (no blank target).
        name_edit.setText("  ")
        name_edit.editingFinished.emit()
        assert item.target_name == "readout_rf_tuned"
        assert name_edit.text() == "readout_rf_tuned"
    finally:
        dialog.close()
    ctrl.set_writeback_item_for_pane.assert_called_once_with(
        "tab-1", "analysis", "ml-1", target_name="readout_rf_tuned"
    )
