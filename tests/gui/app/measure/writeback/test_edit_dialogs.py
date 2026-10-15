"""Writeback editing feedback through the ledger's public Qt surface."""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QDialog, QLabel, QLineEdit, QMessageBox, QPushButton
from zcu_tools.gui.app.measure.writeback import ParameterWriteback, WritebackValues
from zcu_tools.gui.app.measure.writeback.widget import WritebackWidget
from zcu_tools.gui.widgets import error_colour


def button(parent, text: str) -> QPushButton:
    return next(b for b in parent.findChildren(QPushButton) if b.text() == text)


def label_text(parent) -> str:
    return "\n".join(label.text() for label in parent.findChildren(QLabel))


@pytest.fixture
def value_editor(
    qapp, monkeypatch
) -> Iterator[
    tuple[
        WritebackWidget,
        QDialog,
        ParameterWriteback,
        MagicMock,
        list[tuple[object, ...]],
    ]
]:
    item = ParameterWriteback(
        target_name="$resonator.freq",
        description="Resonator frequency",
        proposed_value=6012.345678,
        stderr=0.0213,
    )
    item.session_id = "param-1"
    ctrl = MagicMock()
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = lambda *a, **k: {
        item.session_id: WritebackValues(
            actual_path="Q1.freq",
            current=6011.9,
            proposed=item.proposed_value,
            unit="MHz",
        )
    }
    modal_messages = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *args: modal_messages.append(args)
    )
    widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")
    widget.populate([item], ctrl.writeback_control.get_writeback_values_for_pane())
    widget.show()
    button(widget, "Edit").click()
    dialog = next(d for d in widget.findChildren(QDialog) if d.isVisible())
    yield widget, dialog, item, ctrl, modal_messages
    dialog.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_value_context_and_live_validation(value_editor, qapp):
    _, dialog, _, ctrl, messages = value_editor
    texts = label_text(dialog)
    assert "Resonator frequency" in texts
    assert "Q1.freq" in texts and "Writes to" in texts
    assert "6011.9 MHz" in texts
    assert "0.0213 MHz" in texts
    edits = dialog.findChildren(QLineEdit)
    name = next(e for e in edits if e.text() == "$resonator.freq")
    value = next(e for e in edits if e.text() == "6012.345678")
    value.setText("not a number")
    assert "Enter a number" in label_text(dialog)
    assert not button(dialog, "Save").isEnabled()
    assert not ctrl.set_writeback_item_for_pane.called
    error = next(l for l in dialog.findChildren(QLabel) if "Enter a number" in l.text())
    palette = dialog.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#353535"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))
    dialog.setPalette(palette)
    qapp.processEvents()
    assert error.palette().color(QPalette.ColorRole.WindowText) == error_colour(dialog)
    value.setText("6013 MHz")
    name.setText(" ")
    assert "Apply-as name must not be empty" in label_text(dialog)
    assert not button(dialog, "Save").isEnabled()
    name.setText("$resonator.freq")
    assert button(dialog, "Save").isEnabled()
    assert messages == []


def test_value_matching_unit_save_and_cancel(value_editor):
    _, dialog, item, ctrl, messages = value_editor
    value = next(e for e in dialog.findChildren(QLineEdit) if e.text() == "6012.345678")
    value.setText("6013.25 MHz")
    button(dialog, "Save").click()
    assert not dialog.isVisible()
    assert item.proposed_value == 6013.25
    ctrl.set_writeback_item_for_pane.assert_called_once_with(
        "tab-1",
        "analysis",
        "param-1",
        target_name="$resonator.freq",
        proposed_value=6013.25,
    )
    assert messages == []


def test_value_controller_refusal_stays_inline(value_editor):
    _, dialog, item, ctrl, messages = value_editor
    ctrl.set_writeback_item_for_pane.side_effect = RuntimeError("Tab is busy")
    button(dialog, "Save").click()
    assert "Tab is busy" in label_text(dialog)
    assert dialog.isVisible()
    assert item.proposed_value == 6012.345678
    assert messages == []


@pytest.mark.parametrize(
    ("original", "invalid", "message"),
    [
        (1, "1.2", "Enter an integer"),
        (True, "maybe", "Enter true or false"),
        (1 + 2j, "1+2i", "Enter a complex number"),
    ],
)
def test_value_type_specific_validation(qapp, original, invalid, message):
    item = ParameterWriteback(
        target_name="value", description="Typed value", proposed_value=original
    )
    item.session_id = "param-1"
    widget = WritebackWidget(MagicMock(), tab_id="tab-1")
    widget.populate(
        [item],
        {
            "param-1": WritebackValues(
                actual_path="value", current=None, proposed=original, unit=None
            )
        },
    )
    button(widget, "Edit").click()
    dialog = next(d for d in widget.findChildren(QDialog) if d.isVisible())
    try:
        value = next(
            e for e in dialog.findChildren(QLineEdit) if e.text() == str(original)
        )
        value.setText(invalid)
        assert message in label_text(dialog)
        assert not button(dialog, "Save").isEnabled()
    finally:
        dialog.close()
        widget.deleteLater()
        qapp.processEvents()
