from __future__ import annotations

from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QWidget,
)
from zcu_tools.gui.widgets.dialog_lifecycle import DialogRefStore
from zcu_tools.gui.widgets.dialog_presenter import (
    DestructiveConfirmation,
    QtDialogPresenter,
)


def test_destructive_confirm_opens_non_blocking_message_box(qapp, monkeypatch):
    def fail_exec(_self: QMessageBox) -> int:
        raise AssertionError("destructive_confirm must not call exec()")

    monkeypatch.setattr(QMessageBox, "exec", fail_exec)
    parent = QWidget()
    refs = DialogRefStore()
    presenter = QtDialogPresenter(refs)
    decisions: list[bool] = []

    presenter.destructive_confirm(
        parent,
        DestructiveConfirmation(
            title="Run still stopping",
            message="Force close?",
            action_text="Force Close",
            default=False,
        ),
        on_decision=decisions.append,
    )
    QApplication.processEvents()

    boxes = parent.findChildren(QMessageBox)
    assert len(boxes) == 1
    assert len(refs) == 1
    assert decisions == []

    action_button = next(
        button for button in boxes[0].buttons() if button.text() == "Force Close"
    )
    action_button.click()
    QApplication.processEvents()

    assert decisions == [True]
    assert len(refs) == 0
    parent.deleteLater()
