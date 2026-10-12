"""Real writeback draft editors retain usable exits under remote control."""

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QDialog, QLineEdit, QPushButton
from zcu_tools.gui.app.measure.adapter import ParameterWriteback
from zcu_tools.gui.app.measure.ui.writeback_widget import WritebackWidget
from zcu_tools.gui.control_identity import ControlIdentity


@pytest.mark.parametrize("kind", ["parameter", "waveform"])
@pytest.mark.parametrize("opening", ["existing", "new"])
def test_remote_writeback_editor_keeps_exit_and_does_not_commit_pending_name(
    qapp, waveform_writeback, kind, opening
):
    control, service, waveform_draft = waveform_writeback
    if kind == "parameter":
        control.set_parameter(control.read_context().target, "general.ext.value", 1.0)
        draft = service.create_draft(
            [ParameterWriteback("general.ext.value", "Value", proposed_value=2.0)],
            parameters=control.read_context(),
            experiment_name="test/writeback",
        )
    else:
        draft = waveform_draft
    item = draft.items[0]
    original_name = item.target_name
    ctrl = MagicMock()
    authority = ControlIdentity()
    ctrl.control_identity = authority
    ctrl.get_writeback_applied_for_pane.side_effect = lambda *_: (
        service.get_all_applied(draft)
    )
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = (
        lambda *_, **kwargs: service.preview_values(draft, **kwargs)
    )
    ctrl.set_writeback_item_for_pane.side_effect = lambda _tab, _pane, sid, **kwargs: (
        draft.edit(sid, **kwargs)
    )
    if kind == "waveform":
        ctrl.get_writeback_item_draft_for_pane.return_value = service.get_item_draft(
            draft, item.session_id
        )
    widget = WritebackWidget(ctrl, tab_id="captured", pane="analysis")
    try:
        widget.populate(service.preview_draft(draft), service.preview_values(draft))
        widget.show()
        if opening == "new":
            authority.connect("test", "agent", "owner")
        next(b for b in widget.findChildren(QPushButton) if b.text() == "Edit").click()
        dialog = widget.findChild(QDialog)
        assert dialog is not None and dialog.isModal()
        if opening == "existing":
            if kind == "waveform":
                next(
                    e
                    for e in dialog.findChildren(QLineEdit)
                    if e.text() == original_name
                ).setText("Q1.waveforms.pending")
            authority.connect("test", "agent", "owner")
        assert dialog.isEnabled(), "dialog exit must remain available"
        assert all(not edit.isEnabled() for edit in dialog.findChildren(QLineEdit))
        exit_text = "Cancel" if kind == "parameter" else "Close"
        exit_button = next(
            b for b in dialog.findChildren(QPushButton) if b.text() == exit_text
        )
        assert exit_button.isEnabled()
        exit_button.click()
        assert not dialog.isVisible()
        assert item.target_name == original_name
        assert authority.status().holder == "rpc:agent"
        if kind == "parameter":
            assert item.proposed_value == 2.0
    finally:
        widget.deleteLater()
        qapp.processEvents()
        if kind == "parameter":
            draft.teardown()
