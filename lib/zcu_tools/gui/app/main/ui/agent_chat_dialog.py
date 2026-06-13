"""AgentChatDialog — non-modal dialog for the agent conversation transcript.

Opened by MainWindow via the "Agent…" toolbar button. Displays a scrollable
plain-text transcript of agent tool-call activity, user feedback, and GUI
diagnostics, plus an input field for sending feedback to a blocked agent.

The underlying AgentChatService is the source of truth; this dialog is a pure
View that registers a listener and refreshes on every append. Closing the dialog
does not clear the transcript (service owns the history; re-opening shows it).

Threading: all Qt mutations here are on the main thread. The listener registered
into AgentChatService is called synchronously by that service, which itself is
always called from the main thread — no cross-thread Qt calls occur.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services.agent_chat import AgentChatService

logger = logging.getLogger(__name__)

# Status-label auto-clear delay (milliseconds).
_STATUS_CLEAR_MS = 4000


class AgentChatDialog(QDialog):
    """Non-modal dialog that shows the agent conversation transcript.

    Lifecycle: MainWindow creates an instance on first "Agent…" click, sets
    ``WA_DeleteOnClose`` so Qt destroys it when closed, and clears its own
    reference via a ``finished`` signal. Re-opening creates a fresh instance
    seeded from the service's retained history.
    """

    def __init__(self, ctrl: Controller) -> None:
        super().__init__(None, Qt.WindowType.Window)  # type: ignore[attr-defined]
        self._ctrl = ctrl
        self._chat: AgentChatService = ctrl.get_agent_chat()

        self.setWindowTitle("Agent Chat")
        self.resize(700, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- transcript view ---
        self._transcript = QPlainTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        font = self._transcript.font()
        font.setFamily("Monospace")
        self._transcript.setFont(font)
        layout.addWidget(self._transcript, stretch=1)

        # --- input row ---
        input_row = QHBoxLayout()
        input_row.setSpacing(4)
        self._input = QLineEdit()
        self._input.setPlaceholderText("Type feedback for the agent…")
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input, stretch=1)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._on_send)
        input_row.addWidget(send_btn)
        layout.addLayout(input_row)

        # --- status label ---
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._status_label)

        # Seed with existing transcript history.
        self._refresh_all()

        # Register listener — remove it on close to prevent dangling reference.
        self._chat.add_listener(self._on_transcript_changed)
        self.finished.connect(self._on_finished)

    # ------------------------------------------------------------------
    # Listener & refresh
    # ------------------------------------------------------------------

    def _on_transcript_changed(self) -> None:
        """Called synchronously by AgentChatService on every append (main thread).

        Only appends the newest entry instead of re-rendering the whole buffer,
        so scrollback is not disrupted by intermediate refreshes.
        """
        entries = self._chat.entries()
        if not entries:
            return
        last = entries[-1]
        self._transcript.appendPlainText(self._format_entry(last))

    def _refresh_all(self) -> None:
        """Seed the transcript widget from the full service history."""
        self._transcript.clear()
        for entry in self._chat.entries():
            self._transcript.appendPlainText(self._format_entry(entry))

    @staticmethod
    def _format_entry(entry) -> str:
        """Render one TranscriptEntry as a display line (no timestamp for brevity)."""
        return entry.text

    # ------------------------------------------------------------------
    # Send handler
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        """Post feedback to the inbox and record it in the transcript. Main-thread."""
        text = self._input.text().strip()
        if not text:
            return
        # Wire to the cooperative-interrupt inbox (ADR-0023).
        inbox = self._ctrl.get_feedback_inbox()
        inbox.post(text)
        # Record in the transcript so the user sees their own message.
        self._chat.record_feedback(text)
        self._input.clear()
        # Show delivery status.
        if self._ctrl.has_pending_wait():
            self._status_label.setText("Sent — agent will see it now.")
        else:
            self._status_label.setText("Queued — agent will see it at the next wait.")
        QTimer.singleShot(_STATUS_CLEAR_MS, lambda: self._status_label.setText(""))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_finished(self) -> None:
        """Remove listener when the dialog is closed/destroyed (WA_DeleteOnClose).

        Prevents a stale reference from calling Qt widget methods after deletion.
        """
        self._chat.remove_listener(self._on_transcript_changed)
