"""NotifyUserDialog — non-modal prompt for agent-initiated user questions.

The agent calls gui_notify_user(message, timeout); the dispatch layer opens
this dialog on the main thread via MainWindow.open_notify_prompt. The dialog
is the timeout SSOT (ADR-0025 §dialog-timeout): a QTimer fires here and calls
ctrl.timeout_notify so the notify channel records Timeout rather than relying
on the consumer's backstop to time out independently.

Non-modal (open(), not exec()) so the Qt event loop / RPC socket are never
stalled. WA_DeleteOnClose is set so the widget is freed when closed.

Thread-contract: all methods run on the main thread (Qt slots / QTimer callback
/ dialog accept/reject). The consumer (ctrl.await_notify) is off-main.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller


class NotifyUserDialog(QDialog):
    """Non-modal dialog showing an agent message and waiting for user response.

    Closes itself on Reply, Dismiss, window-X, or QTimer expiry. Each path
    calls exactly one of ctrl.reply_notify / dismiss_notify / timeout_notify
    (the set-once latch in NotifyChannel makes multi-fire safe, but the dialog
    guards against it with _closed so no extra calls reach the controller).
    """

    def __init__(
        self,
        token: int,
        message: str,
        timeout: float,
        controller: Controller,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._token = token
        self._ctrl = controller
        self._closed = False  # guard against double-fire (QTimer + user action)

        self.setWindowTitle("Agent prompt")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # Non-modal: open() later; not exec() which would block the event loop.
        self.setModal(False)
        self.resize(480, 200)

        layout = QVBoxLayout(self)

        lbl = QLabel(message)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(lbl)

        self._reply_edit = QLineEdit()
        self._reply_edit.setPlaceholderText("Type your reply here (optional)…")
        layout.addWidget(self._reply_edit)

        # Reply + Dismiss buttons in a standard button box.
        btn_box = QDialogButtonBox()
        reply_btn = btn_box.addButton("Reply", QDialogButtonBox.ButtonRole.AcceptRole)
        dismiss_btn = btn_box.addButton(
            "Dismiss", QDialogButtonBox.ButtonRole.RejectRole
        )
        assert reply_btn is not None  # addButton always returns a button for text+role
        assert dismiss_btn is not None
        reply_btn.clicked.connect(self._on_reply)
        dismiss_btn.clicked.connect(self._on_dismiss)
        layout.addWidget(btn_box)

        # QTimer is the timeout SSOT (ADR-0025): fires after `timeout` seconds
        # and calls ctrl.timeout_notify so the channel records Timeout, not Dismiss.
        # singleShot → fires once; timeout is in milliseconds.
        self._timer: QTimer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer)  # type: ignore[attr-defined]
        if timeout > 0:
            self._timer.start(int(timeout * 1000))

    # ------------------------------------------------------------------
    # Slot implementations (main thread)
    # ------------------------------------------------------------------

    def _on_reply(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        text = self._reply_edit.text()  # may be an empty string — valid reply
        self._ctrl.reply_notify(self._token, text)
        self.close()

    def _on_dismiss(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._ctrl.dismiss_notify(self._token)
        self.close()

    def _on_timer(self) -> None:
        """QTimer expiry — dialog is the timeout SSOT."""
        if self._closed:
            return
        self._closed = True
        self._ctrl.timeout_notify(self._token)
        self.close()

    def closeEvent(self, event: object) -> None:  # type: ignore[override]
        """Window-X close → treated as Dismiss (user explicitly closed without answering)."""
        if not self._closed:
            self._closed = True
            self._timer.stop()
            self._ctrl.dismiss_notify(self._token)
        super().closeEvent(event)  # type: ignore[misc]
