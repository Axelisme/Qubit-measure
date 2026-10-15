"""Observable decisions and labels of the Qt dialog presenter."""

import pytest
from qtpy.QtWidgets import QMessageBox, QWidget
from zcu_tools.gui.widgets.dialog_presenter import QtDialogPresenter


@pytest.mark.parametrize("decision", ["action", "retry", "dismiss"])
def test_destructive_choice_only_accepts_explicit_action(qapp, decision):
    parent = QWidget()
    presenter = QtDialogPresenter()
    decisions: list[bool] = []
    try:
        presenter.destructive_confirm(
            parent,
            "Background work",
            "Work is still running.",
            action_text="Force close",
            cancel_text="Wait again",
            on_decision=decisions.append,
        )
        box = parent.findChild(QMessageBox)
        assert box is not None
        buttons = {button.text(): button for button in box.buttons()}
        assert set(buttons) == {"Force close", "Wait again"}
        assert box.defaultButton() is buttons["Wait again"]
        assert decisions == []
        if decision == "dismiss":
            box.reject()
        else:
            buttons["Force close" if decision == "action" else "Wait again"].click()
        qapp.processEvents()
        assert decisions == [decision == "action"]
    finally:
        parent.deleteLater()
        qapp.processEvents()
