"""Retained non-modal exception reports preserve the message and traceback."""

from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox
from zcu_tools.gui.app.measure.ui.error_handler import show_error_dialog


def test_show_error_dialog_retains_nonmodal_message_and_traceback(qapp):
    # The old modal presenter must unwind too, so red cannot hang in exec().
    def dismiss_modal() -> None:
        modal = qapp.activeModalWidget()
        if modal is not None:
            modal.close()

    QTimer.singleShot(0, dismiss_modal)
    try:
        raise ValueError("test error")
    except ValueError as error:
        show_error_dialog(type(error), error, error.__traceback__)
    qapp.processEvents()
    boxes = [
        w
        for w in qapp.topLevelWidgets()
        if isinstance(w, QMessageBox) and w.isVisible()
    ]
    try:
        assert len(boxes) == 1, (
            "the report must remain visible after presentation returns"
        )
        box = boxes[0]
        assert not box.isModal()
        assert box.text() == "ValueError: test error"
        assert "raise ValueError" in box.detailedText()
        assert "ValueError: test error" in box.detailedText()
    finally:
        for box in boxes:
            box.close()
        qapp.processEvents()
