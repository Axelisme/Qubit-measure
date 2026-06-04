from unittest.mock import MagicMock, patch

from qtpy.QtWidgets import QCheckBox, QDialog, QLineEdit, QPushButton
from zcu_tools.gui.app.main.adapter import MetaDictWriteback, ModuleWriteback
from zcu_tools.gui.app.main.ui.writeback_widget import WritebackWidget


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


def test_edit_md_item_can_change_target_name(qapp):
    """The metadict Edit dialog exposes an editable apply-as (target_name)."""
    item = MetaDictWriteback(
        target_name="r_f", description="freq", proposed_value=6000.0
    )
    item.session_id = "md-1"

    widget = WritebackWidget(MagicMock())
    widget.populate([item])

    cb = QCheckBox()
    widget._edit_md_item(item, cb)
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
        assert item.session_id == "md-1"
    finally:
        dialog.close()


def test_edit_md_item_rejects_blank_target_name(qapp):
    item = MetaDictWriteback(
        target_name="r_f", description="freq", proposed_value=6000.0
    )
    item.session_id = "md-1"

    widget = WritebackWidget(MagicMock())
    widget.populate([item])

    cb = QCheckBox()
    widget._edit_md_item(item, cb)
    dialog = _find_open_dialog("Edit Value:")
    try:
        name_edit = dialog.findChildren(QLineEdit)[0]
        name_edit.setText("   ")
        save_btn = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == "Save"
        )
        # The validation error pops a modal QMessageBox; stub it so the test
        # does not block on it.
        with patch("zcu_tools.gui.app.main.ui.writeback_widget.QMessageBox.critical"):
            save_btn.click()

        # Blank name is rejected: target_name unchanged, dialog stays open.
        assert item.target_name == "r_f"
    finally:
        dialog.close()


def test_edit_cfg_item_can_change_target_name(qapp):
    """The module/waveform Edit dialog exposes an editable apply-as that commits
    on editingFinished."""
    item = ModuleWriteback(
        target_name="readout_rf", description="A module", edit_schema=MagicMock()
    )
    item.session_id = "ml-1"
    item.editor_id = "editor-9"

    ctrl = MagicMock()
    ctrl.get_cfg_editor_root.return_value = MagicMock()
    widget = WritebackWidget(ctrl)
    widget.populate([item])

    cb = QCheckBox()
    widget._edit_cfg_item(item, cb)
    dialog = _find_open_dialog("Edit Config:")
    try:
        name_edit = dialog.findChildren(QLineEdit)[0]
        assert name_edit.text() == "readout_rf"

        name_edit.setText("readout_rf_tuned")
        name_edit.editingFinished.emit()
        assert item.target_name == "readout_rf_tuned"

        # Blank reverts to the previous name (no blank target).
        name_edit.setText("  ")
        name_edit.editingFinished.emit()
        assert item.target_name == "readout_rf_tuned"
        assert name_edit.text() == "readout_rf_tuned"
    finally:
        dialog.close()
