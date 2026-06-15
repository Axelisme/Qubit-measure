"""FloatingFeedbackWidget — corner overlay for user→agent feedback during ops.

Floats in the MainWindow's bottom-right corner (absolute positioning, no
layout manager). Visible only while at least one live operation is in progress
(MainWindow drives show/hide via _refresh_feedback_widget()).

The widget is app-level (parent = MainWindow), not tab-level, so it persists
across tab switches and represents the single foreground operation.

Stop-gating: 'Send & Stop' is enabled only when the active operation has a
cancel hook registered (ADR-0025 §Stop-gating). Gating is refreshed by
MainWindow each time the op count or op type changes.

NOTE: future enhancement — could gate display further on 'MCP client
connected' (i.e. an agent is driving), but Stage 4a uses C1 (always visible
when any op is live) because the GUI has no reliable agent-session signal.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller

logger = logging.getLogger(__name__)

# Visual constants
_WIDGET_WIDTH = 320
_WIDGET_MARGIN = 12  # gap from the window's right/bottom edges


class FloatingFeedbackWidget(QWidget):
    """Bottom-right floating overlay for sending feedback to the active op.

    Public API (called by MainWindow):
    - refresh_gating(): re-read can_cancel_active_operation() and
      enable/disable 'Send & Stop' accordingly.
    - clear_input(): wipe the text field (called on hide).
    """

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl

        # Frameless, always on top of other child widgets.
        self.setWindowFlags(Qt.SubWindow)  # type: ignore[attr-defined]
        self.setAttribute(Qt.WA_StyledBackground, True)  # type: ignore[attr-defined]
        self.setStyleSheet(
            "FloatingFeedbackWidget {"
            "  background-color: rgba(245, 245, 245, 230);"
            "  border: 1px solid #aaa;"
            "  border-radius: 6px;"
            "}"
        )
        self.setFixedWidth(_WIDGET_WIDTH)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        title = QLabel("Send to agent")
        title.setStyleSheet("font-weight: bold; font-size: 11px; color: #333;")
        root.addWidget(title)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Message…")
        self._input.textChanged.connect(self._on_text_changed)
        root.addWidget(self._input)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._send_btn = QPushButton("Send")
        self._send_btn.setToolTip("Nudge the agent (operation continues)")
        self._send_btn.clicked.connect(self._on_send_clicked)
        btn_row.addWidget(self._send_btn)

        self._stop_btn = QPushButton("Send & Stop")
        self._stop_btn.setToolTip("Send this message and cancel the active operation")
        self._stop_btn.clicked.connect(self._on_send_stop_clicked)
        btn_row.addWidget(self._stop_btn)

        root.addLayout(btn_row)
        self.adjustSize()

        # Initialise button states.
        self._on_text_changed(self._input.text())
        self.refresh_gating()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def refresh_gating(self) -> None:
        """Re-evaluate whether 'Send & Stop' should be enabled.

        Reads Controller.can_cancel_active_operation(); ops without a cancel
        hook (connect / FIT-analyze / device connect-disconnect) disable Stop.
        """
        can_stop = self._ctrl.can_cancel_active_operation()
        self._stop_btn.setEnabled(can_stop and bool(self._input.text().strip()))

    def clear_input(self) -> None:
        """Clear the text field (called when the widget is hidden)."""
        self._input.clear()

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_text_changed(self, text: str) -> None:
        has_text = bool(text.strip())
        # Send requires non-blank text (OperationChannel.message ignores blank).
        self._send_btn.setEnabled(has_text)
        # Stop gating: also re-read can_cancel so the button stays correct
        # if the op type changed while the user was typing.
        can_stop = self._ctrl.can_cancel_active_operation()
        self._stop_btn.setEnabled(has_text and can_stop)

    def _on_send_clicked(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        logger.debug("FloatingFeedbackWidget: send nudge %r", text)
        self._ctrl.send_feedback(text, stop=False)
        self._input.clear()

    def _on_send_stop_clicked(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        logger.debug("FloatingFeedbackWidget: send+stop %r", text)
        self._ctrl.send_feedback(text, stop=True)
        self._input.clear()
