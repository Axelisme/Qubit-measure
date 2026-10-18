"""Retained non-modal exception reports preserve the message and traceback."""

from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QMessageBox, QWidget
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


def test_error_report_does_not_keep_application_alive_after_primary_close(
    qapp: QApplication,
) -> None:
    window = QWidget()
    window.show()
    error = RuntimeError("shutdown with an open error report")
    show_error_dialog(type(error), error, None)
    reports = [
        widget
        for widget in qapp.topLevelWidgets()
        if isinstance(widget, QMessageBox) and widget.isVisible()
    ]
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: qapp.exit(72))
    try:
        assert len(reports) == 1
        assert not reports[0].isModal()
        # Closing the primary window must end the application event loop.
        # The watchdog only turns the missing exit into an assertion failure.
        QTimer.singleShot(0, window.close)
        watchdog.start(10_000)
        exit_code = qapp.exec()
        assert not window.isVisible()
        assert exit_code == 0, "the retained error report prevented normal Qt exit"
    finally:
        watchdog.stop()
        for widget in qapp.topLevelWidgets():
            if isinstance(widget, QMessageBox):
                widget.close()
        window.close()
        window.deleteLater()
        qapp.processEvents()
