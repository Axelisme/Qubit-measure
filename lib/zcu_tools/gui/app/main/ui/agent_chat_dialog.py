"""AgentChatDialog — non-modal dialog for the agent conversation transcript.

Opened by MainWindow via the "Agent…" toolbar button. Displays a scrollable
plain-text transcript of agent tool-call activity, user feedback, GUI
diagnostics, AND the embedded claude child's stream-json output (B0).

B0 additions
------------
- **Start** button: opens an optional task-prompt dialog, then spawns the
  embedded ``claude`` child via the backend obtained from
  ``Controller.get_agent_session()``.
- **Stop** button: sends SIGINT to the child.
- **Status bar**: shows idle / working / waiting / stopped plus the last
  result cost (optional).
- **Input routing**: ``Send`` routes text depending on ``AgentState``:
    - idle or stopped → queued into the FeedbackInbox (unchanged Phase-A path)
    - working → sent as stdin message to the running claude child
    - waiting (has_pending_wait) → sent as FeedbackInbox feedback wakeup

The underlying AgentChatService is the source of truth; this dialog is a pure
View that registers a listener and refreshes on every append. Closing the dialog
does not clear the transcript (service owns the history; re-opening shows it).

Threading: all Qt mutations here are on the main thread. The listener registered
into AgentChatService is called synchronously by that service, which itself is
always called from the main thread — no cross-thread Qt calls occur.

B1a: the dialog depends only on ``AgentSessionPort`` (Qt-free control surface)
and never imports the concrete ``AgentRunner``. The Controller factory
``get_agent_session()`` injects the CLI/subscription backend; a future API-mode
backend would implement the same port without any dialog changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services.agent_chat import AgentChatService
    from zcu_tools.gui.app.main.services.ports import AgentSessionPort, AgentState

logger = logging.getLogger(__name__)

# Status-label auto-clear delay (milliseconds).
_STATUS_CLEAR_MS = 4000

# Default task prompt shown in the Start dialog.
_DEFAULT_TASK = (
    "Check the current state of the measure-gui and report what is configured."
)


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

        # B1a: session backend obtained lazily from the Controller factory.
        # The type is AgentSessionPort (Qt-free); no concrete AgentRunner import.
        self._session: AgentSessionPort | None = None

        self.setWindowTitle("Agent Chat")
        self.resize(700, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- runner control row ---
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)
        self._start_btn = QPushButton("Start")
        self._start_btn.setToolTip("Spawn an embedded claude agent to operate this GUI")
        self._start_btn.clicked.connect(self._on_start)
        ctrl_row.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setToolTip("Send SIGINT to the running agent (graceful stop)")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self._stop_btn)
        self._agent_status = QLabel("idle")
        self._agent_status.setStyleSheet("color: gray; font-style: italic;")
        ctrl_row.addWidget(self._agent_status, stretch=1)
        layout.addLayout(ctrl_row)

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
        self._input.setPlaceholderText(
            "Type a message (routed to agent stdin / feedback inbox / queue)…"
        )
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
    def _format_entry(entry) -> str:  # type: ignore[no-untyped-def]
        """Render one TranscriptEntry as a display line (no timestamp for brevity)."""
        return entry.text

    # ------------------------------------------------------------------
    # B0/B1a — Start / Stop
    # ------------------------------------------------------------------

    def _ensure_session(self) -> AgentSessionPort:
        """Return the session backend, creating it on first call.

        Registers ``_update_runner_ui`` as a state listener so the dialog
        refreshes without any callback rewrap hack (B1a).
        """
        if self._session is None:
            self._session = self._ctrl.get_agent_session()
            # Subscribe to state changes; the listener holds a reference to self
            # which is fine — the dialog owns the session lifetime and unregisters
            # on close.
            self._session.add_state_listener(self._update_runner_ui)
        return self._session

    def _on_start(self) -> None:
        """Prompt for a task and spawn the embedded claude child."""
        task, ok = QInputDialog.getText(
            self,
            "Start Agent",
            "Task for the agent:",
            QLineEdit.EchoMode.Normal,
            _DEFAULT_TASK,
        )
        if not ok or not task.strip():
            return
        task = task.strip()

        session = self._ensure_session()
        repo_root = self._ctrl.get_project_root()
        session.start(task, repo_root)
        # Record a local transcript note that a new session was started.
        self._chat.record_feedback(f"[start] {task}")

    def _on_stop(self) -> None:
        """Send SIGINT to the running agent."""
        if self._session is not None:
            self._session.stop()

    def _update_runner_ui(self, state: AgentState) -> None:
        """Refresh the Start/Stop buttons and status label for the new state."""
        is_active = state in ("working", "waiting")
        self._start_btn.setEnabled(not is_active)
        self._stop_btn.setEnabled(is_active)
        label_map = {
            "idle": "idle",
            "working": "working…",
            "waiting": "waiting for feedback…",
            "stopped": "stopped",
        }
        self._agent_status.setText(label_map.get(state, state))

    # ------------------------------------------------------------------
    # Send handler
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        """Single smart entry: Send always does the right thing. Main-thread.

        - No live agent process → start a new turn with this text (so the input
          box alone drives the conversation, Claude-Code style; no separate Start
          needed).
        - Live process blocked on an operation → wake it via the feedback inbox
          (ADR-0023 cooperative interrupt).
        - Live process otherwise → next user turn via stdin.
        """
        text = self._input.text().strip()
        if not text:
            return

        session = self._session

        # No live process → start a new turn/session with this text.
        if session is None or not session.is_running():
            session = self._ensure_session()
            repo_root = self._ctrl.get_project_root()
            session.start(text, repo_root)
            self._chat.record_feedback(text)
            self._input.clear()
            self._set_status("Started agent.")
            return

        # Live process blocked on an operation → wake via feedback inbox.
        if session.state == "waiting" or self._ctrl.has_pending_wait():
            self._ctrl.get_feedback_inbox().post(text)
            self._chat.record_feedback(text)
            self._input.clear()
            self._set_status("Sent — agent will see it now.")
            return

        # Live process between/within turns → next user turn via stdin.
        session.send_user_message(text)
        self._chat.record_feedback(text)
        self._input.clear()
        self._set_status("Sent to agent.")
        self._input.clear()
        self._set_status("Queued — agent will see it at the next wait.")

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        QTimer.singleShot(_STATUS_CLEAR_MS, lambda: self._status_label.setText(""))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_finished(self) -> None:
        """Remove listener when the dialog is closed/destroyed (WA_DeleteOnClose).

        Prevents a stale reference from calling Qt widget methods after deletion.
        If the session is still active, stop it so the child process does not
        outlive the dialog.
        """
        self._chat.remove_listener(self._on_transcript_changed)
        if self._session is not None and self._session.state in ("working", "waiting"):
            self._session.stop()
