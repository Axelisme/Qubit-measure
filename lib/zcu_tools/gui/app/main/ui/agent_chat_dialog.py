"""AgentChatDialog — session picker + conversation dialog for the embedded agent.

B1b-2 structure
---------------
The dialog uses a ``QStackedWidget`` with two pages:

  Page 0 — **Picker**: lists all known agent sessions from the registry.
    Each row shows title / status / created and has per-row action buttons.
    A **New** button starts a blank conversation (does NOT spawn immediately;
    the first Send does it, decision E).

  Page 1 — **Conversation**: transcript display + input box + **Send** + **Close**.
    - Send routes based on agent state (three-way logic, unchanged from B1a).
    - Close = ``session.detach()`` (Independent) or ``session.stop()`` (CLI).
      Returns to Picker and refreshes the list.  Does NOT kill the session.

Backend mode (decision F)
--------------------------
  ``ctrl.agent_backend_mode()`` is read once at construction:
    - ``"independent"`` (default): picker is shown first; Close=detach.
    - ``"cli"``: no picker; starts directly on Conversation page; Close=stop
      (same as the old B1a behaviour).

Threading
---------
  All Qt mutations are on the main thread.  Listener registered into
  AgentChatService is called synchronously (also main thread).

B1a invariant preserved
-----------------------
  The dialog depends only on ``AgentSessionPort`` (Qt-free); no concrete
  ``AgentRunner`` or ``IndependentAgentSession`` import occurs here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtGui import (  # type: ignore[attr-defined]
    QColor,
    QTextCharFormat,
    QTextCursor,
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services.agent_chat import AgentChatService
    from zcu_tools.gui.app.main.services.agent_session_registry import (
        AgentSessionRecord,
    )
    from zcu_tools.gui.app.main.services.ports import AgentSessionPort, AgentState

logger = logging.getLogger(__name__)

# Status-label auto-clear delay (milliseconds).
_STATUS_CLEAR_MS = 4000

# Page indices in the QStackedWidget.
_PAGE_PICKER = 0
_PAGE_CONVERSATION = 1

# Maximum characters of task shown as session title in the picker.
_TITLE_MAX_CHARS = 60

# ---------------------------------------------------------------------------
# Transcript entry formatting — kind → (hex_color, line_prefix)
#
# Colors chosen for readability on both light and dark Qt themes:
#   - Avoid pure black/white (clashes with theme background).
#   - Use medium-saturation hues so they are legible on dark *and* light bg.
# ---------------------------------------------------------------------------

# Mapping from TranscriptEntry.kind to (color_hex, line_prefix).
# "feedback" text already starts with "you: " (set by AgentChatService); the
# "▶ " prefix is prepended to give a visual anchor without duplicating "you".
_KIND_FORMAT: dict[str, tuple[str, str]] = {
    "feedback": ("#4d9ff0", "▶ "),  # blue — user input
    "assistant": ("#4ec94e", "◀ "),  # green — agent prose
    "tool_use": ("#888888", "  "),  # grey — secondary info
    "tool_result": ("#888888", "  "),  # grey — secondary info
    "system": ("#888888", "  "),  # grey — secondary info
    "result": ("#aaaaaa", "  "),  # lighter grey — done marker
    "activity": ("#888888", "  "),  # grey — MCP activity
    "diagnostic": ("#e87c30", "! "),  # orange — warnings / errors
}
_KIND_FORMAT_DEFAULT: tuple[str, str] = ("#aaaaaa", "  ")


def entry_format(kind: str) -> tuple[str, str]:
    """Return (color_hex, prefix) for a given TranscriptEntry kind.

    Pure function — no Qt dependency — so it can be unit-tested without a QApp.
    """
    return _KIND_FORMAT.get(kind, _KIND_FORMAT_DEFAULT)


# ---------------------------------------------------------------------------
# HistoryLineEdit — QLineEdit with shell-style Up/Down history navigation
# ---------------------------------------------------------------------------


class HistoryLineEdit(QLineEdit):
    """QLineEdit with Up/Down key navigation through submitted message history.

    Navigation semantics (standard shell behaviour):
      - Up: move to the previous (older) entry.  Stops at the oldest.
      - Down: move to the next (newer) entry.  Past the newest, restores the
        draft that was in the field when Up was first pressed.
      - The draft is saved on the *first* Up keypress of a navigation sequence
        and restored when Down goes past the newest history entry.

    Callers must call ``push_history(text)`` after a successful send so the
    submitted text is appended.  Empty strings and consecutive duplicates are
    not stored.

    Only Up/Down are overridden; all other keys (including Left/Right/cursor
    movement) pass through unchanged.  Because QLineEdit is single-line, Up
    and Down have no built-in meaning, so intercepting them is safe.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history: list[str] = []
        # _index == len(_history) means "not navigating / at the draft position".
        self._index: int = 0
        # Draft saved when the user first presses Up.
        self._draft: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_history(self, text: str) -> None:
        """Append *text* to history after a successful send.

        Empty strings are ignored.  Consecutive duplicates of the last entry
        are also ignored (avoids accumulating the same message on rapid resend).
        """
        if not text:
            return
        if self._history and self._history[-1] == text:
            return
        self._history.append(text)
        # Reset navigation state so next Up starts from the newest entry.
        self._index = len(self._history)
        self._draft = ""

    # ------------------------------------------------------------------
    # Key event override
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()

        if key == Qt.Key.Key_Up:  # type: ignore[attr-defined]
            self._navigate_up()
            event.accept()
            return

        if key == Qt.Key.Key_Down:  # type: ignore[attr-defined]
            self._navigate_down()
            event.accept()
            return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _navigate_up(self) -> None:
        """Move one step toward older history entries."""
        if not self._history:
            return

        if self._index == len(self._history):
            # First Up press: save current text as draft before overwriting.
            self._draft = self.text()

        if self._index > 0:
            self._index -= 1
            self.setText(self._history[self._index])

    def _navigate_down(self) -> None:
        """Move one step toward newer history entries (or restore draft)."""
        if self._index < len(self._history):
            self._index += 1
            if self._index == len(self._history):
                # Past the newest entry: restore the saved draft.
                self.setText(self._draft)
            else:
                self.setText(self._history[self._index])


class AgentChatDialog(QDialog):
    """Non-modal dialog for the embedded agent conversation.

    Lifecycle (independent mode, default):
      - showEvent triggers a picker refresh.
      - User clicks New → switch to conversation page with cleared transcript.
        No process is spawned yet.
      - User types and sends → ``_on_send`` is triggered; if no live session,
        ``session.start(text, repo_root)`` is called (first Send = task).
      - User clicks Attach/Resume → ``ctrl.attach_agent_session(record)`` builds
        a session that tail-replays the existing log; switch to conversation.
      - User clicks Close (in conversation) → ``session.detach()`` (stops tail,
        NOT the supervisor), clear transcript, switch to picker, refresh.
      - User clicks Stop-Remove (in picker) → ``ctrl.remove_agent_session(id)``,
        refresh list.

    Lifecycle (CLI mode, ``ZCU_AGENT_BACKEND=cli``):
      - No picker; dialog starts directly on conversation page.
      - Close = ``session.stop()`` (child process dies with the dialog).
    """

    def __init__(self, ctrl: Controller) -> None:
        super().__init__(None, Qt.WindowType.Window)  # type: ignore[attr-defined]
        self._ctrl = ctrl
        self._chat: AgentChatService = ctrl.get_agent_chat()

        # Backend returned from Controller factory; None until a session is chosen.
        self._session: AgentSessionPort | None = None

        # Read mode once so the picker vs. no-picker decision is stable.
        self._cli_mode: bool = ctrl.agent_backend_mode() == "cli"

        self.setWindowTitle("Agent Chat")
        self.resize(700, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # QStackedWidget holding picker (page 0) and conversation (page 1).
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Build pages.
        self._stack.addWidget(self._build_picker_page())
        self._stack.addWidget(self._build_conversation_page())

        # Register transcript listener — removed on close.
        self._chat.add_listener(self._on_transcript_changed)
        self.finished.connect(self._on_finished)

        if self._cli_mode:
            # CLI mode: skip picker, start directly on conversation page.
            # A fresh CLI session is built eagerly (same as old B1a behaviour).
            self._session = self._ctrl.new_agent_session()
            self._session.add_state_listener(self._update_runner_ui)
            self._stack.setCurrentIndex(_PAGE_CONVERSATION)
            self._refresh_all()
        else:
            # Independent mode: show picker first.
            self._stack.setCurrentIndex(_PAGE_PICKER)

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------

    def _build_picker_page(self) -> QWidget:
        """Build the session-picker page (index 0)."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header row.
        header = QHBoxLayout()
        header.setSpacing(4)
        title_label = QLabel("Agent Sessions")
        title_label.setStyleSheet("font-weight: bold;")
        header.addWidget(title_label, stretch=1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Reload the session list from the registry")
        refresh_btn.clicked.connect(self._refresh_picker)
        header.addWidget(refresh_btn)
        new_btn = QPushButton("New")
        new_btn.setToolTip("Start a blank conversation (first Send becomes the task)")
        new_btn.clicked.connect(self._on_picker_new)
        header.addWidget(new_btn)
        layout.addLayout(header)

        # Session list.
        self._picker_list = QListWidget()
        self._picker_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection  # type: ignore[attr-defined]
        )
        layout.addWidget(self._picker_list, stretch=1)

        # Per-row action buttons (operate on the selected row).
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        attach_btn = QPushButton("Attach / Resume")
        attach_btn.setToolTip(
            "Attach to the selected session (tail its log; running=live, stopped=read-only history)"
        )
        attach_btn.clicked.connect(self._on_picker_attach)
        btn_row.addWidget(attach_btn)
        stop_remove_btn = QPushButton("Stop / Remove")
        stop_remove_btn.setToolTip(
            "Running: stop the supervisor then remove; Stopped: just remove the record"
        )
        stop_remove_btn.clicked.connect(self._on_picker_stop_remove)
        btn_row.addWidget(stop_remove_btn)
        layout.addLayout(btn_row)

        return page

    def _build_conversation_page(self) -> QWidget:
        """Build the conversation page (index 1)."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Agent-status / header row (shown in non-CLI mode; reused in CLI mode).
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)
        self._agent_status = QLabel("idle")
        self._agent_status.setStyleSheet("color: gray; font-style: italic;")
        ctrl_row.addWidget(self._agent_status, stretch=1)
        layout.addLayout(ctrl_row)

        # Transcript.
        self._transcript = QPlainTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth  # type: ignore[attr-defined]
        )
        font = self._transcript.font()
        font.setFamily("Monospace")
        self._transcript.setFont(font)
        layout.addWidget(self._transcript, stretch=1)

        # Input row — HistoryLineEdit enables Up/Down shell-style history.
        input_row = QHBoxLayout()
        input_row.setSpacing(4)
        self._input: HistoryLineEdit = HistoryLineEdit()
        self._input.setPlaceholderText(
            "Type a message (routed to agent stdin / feedback inbox)…"
        )
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input, stretch=1)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._on_send)
        input_row.addWidget(send_btn)
        layout.addLayout(input_row)

        # Close button + status label row.
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(4)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; font-style: italic;")
        bottom_row.addWidget(self._status_label, stretch=1)
        close_btn = QPushButton("Close" if not self._cli_mode else "Stop")
        close_btn.setToolTip(
            "Detach from this session (keeps the agent running)"
            if not self._cli_mode
            else "Stop the agent and close"
        )
        close_btn.clicked.connect(self._on_conversation_close)
        bottom_row.addWidget(close_btn)
        layout.addLayout(bottom_row)

        return page

    # ------------------------------------------------------------------
    # showEvent — auto-refresh picker when dialog is (re-)shown
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._cli_mode and self._stack.currentIndex() == _PAGE_PICKER:
            self._refresh_picker()

    # ------------------------------------------------------------------
    # Picker page actions
    # ------------------------------------------------------------------

    def _refresh_picker(self) -> None:
        """Reload the session list from the registry."""
        self._picker_list.clear()
        records = self._ctrl.list_agent_sessions()
        for rec in records:
            status = rec.get("status", "?")
            title = rec.get("title", "")[:_TITLE_MAX_CHARS]
            created = rec.get("created", "")[:19]  # truncate to date+time
            label = f"[{status}] {title}  ({created})"
            item = QListWidgetItem(label)
            # Store the full record on the item for retrieval.
            item.setData(Qt.ItemDataRole.UserRole, rec)  # type: ignore[attr-defined]
            self._picker_list.addItem(item)

    def _selected_record(self) -> AgentSessionRecord | None:
        """Return the record for the currently selected picker row, or None."""
        items = self._picker_list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]

    def _on_picker_new(self) -> None:
        """Start a blank conversation (no process spawned yet; decision E)."""
        self._switch_to_conversation(session=self._ctrl.new_agent_session())

    def _on_picker_attach(self) -> None:
        """Attach to (or resume) the selected session."""
        rec = self._selected_record()
        if rec is None:
            self._set_status("Select a session first.")
            return
        session = self._ctrl.attach_agent_session(rec)
        self._switch_to_conversation(session=session)

    def _on_picker_stop_remove(self) -> None:
        """Stop (if running) and remove the selected session record (decision C)."""
        rec = self._selected_record()
        if rec is None:
            self._set_status("Select a session first.")
            return
        session_id = rec.get("session_id", "")
        self._ctrl.remove_agent_session(session_id)
        self._refresh_picker()

    # ------------------------------------------------------------------
    # Conversation page actions
    # ------------------------------------------------------------------

    def _on_conversation_close(self) -> None:
        """Close button on conversation page.

        Independent mode: detach (stops tail, NOT supervisor); return to picker.
        CLI mode: stop the session (kills the child process).
        """
        if self._session is not None:
            self._session.detach()
        self._session = None

        if self._cli_mode:
            # In CLI mode Close = stop; the dialog can be re-opened with a new session.
            pass
        else:
            # Return to picker and refresh the list so the new status shows.
            self._chat.clear()
            self._transcript.clear()
            self._stack.setCurrentIndex(_PAGE_PICKER)
            self._refresh_picker()

    def _switch_to_conversation(self, session: AgentSessionPort) -> None:
        """Detach any previous session, set the new one, switch to conversation page."""
        if self._session is not None:
            self._session.detach()

        self._session = session
        self._session.add_state_listener(self._update_runner_ui)

        # Clear transcript display; the new session's attach() will replay history.
        self._chat.clear()
        self._transcript.clear()
        self._refresh_all()

        self._stack.setCurrentIndex(_PAGE_CONVERSATION)
        self._update_runner_ui(session.state)

    # ------------------------------------------------------------------
    # Listener & refresh
    # ------------------------------------------------------------------

    def _on_transcript_changed(self) -> None:
        """Append the newest transcript entry with colour (main thread, synchronous)."""
        entries = self._chat.entries()
        if not entries:
            return
        self._append_colored_entry(entries[-1])

    def _refresh_all(self) -> None:
        """Seed the transcript widget from the full service history (with colour)."""
        self._transcript.clear()
        for entry in self._chat.entries():
            self._append_colored_entry(entry)

    def _append_colored_entry(self, entry) -> None:  # type: ignore[no-untyped-def]
        """Insert one transcript entry as a coloured line at the end.

        Uses QTextCursor + QTextCharFormat so each entry can carry its own
        foreground colour while QPlainTextEdit remains read-only and scrollable.
        After insertion the view auto-scrolls to the bottom.
        """
        color_hex, prefix = entry_format(entry.kind)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color_hex))

        cursor: QTextCursor = self._transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)  # type: ignore[attr-defined]

        # Insert a newline separator before each entry (except the very first).
        doc = self._transcript.document()
        if doc is not None and not doc.isEmpty():
            cursor.insertText("\n", QTextCharFormat())

        cursor.insertText(f"{prefix}{entry.text}", fmt)

        # Scroll to the bottom so the newest entry is visible.
        self._transcript.setTextCursor(cursor)
        self._transcript.ensureCursorVisible()

    # ------------------------------------------------------------------
    # Send handler (three-state routing, unchanged from B1a)
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        """Single smart entry point for the Send button / Return key.

        - No live session → start a new turn with this text (decision E: first
          Send is the task for the session).
        - Live session blocked on an operation → wake via feedback inbox (ADR-0023).
        - Live session between turns → next user turn via stdin / spool.
        """
        text = self._input.text().strip()
        if not text:
            return

        session = self._session

        # No live process → start a new turn/session with this text.
        if session is None or not session.is_running():
            if session is None:
                session = self._ctrl.new_agent_session()
                self._session = session
                session.add_state_listener(self._update_runner_ui)
            repo_root = self._ctrl.get_project_root()
            session.start(self._ctrl.build_first_turn_task(text), repo_root)
            self._chat.record_feedback(text)
            self._input.push_history(text)
            self._input.clear()
            self._set_status("Started agent.")
            return

        # Live process blocked on an operation → wake via feedback inbox.
        if session.state == "waiting" or self._ctrl.has_pending_wait():
            self._ctrl.get_feedback_inbox().post(text)
            self._chat.record_feedback(text)
            self._input.push_history(text)
            self._input.clear()
            self._set_status("Sent — agent will see it now.")
            return

        # Live process between/within turns → next user turn via stdin.
        session.send_user_message(text)
        self._chat.record_feedback(text)
        self._input.push_history(text)
        self._input.clear()
        self._set_status("Sent to agent.")

    # ------------------------------------------------------------------
    # Agent-state UI
    # ------------------------------------------------------------------

    def _update_runner_ui(self, state: AgentState) -> None:
        """Refresh the status label for the new agent state."""
        label_map = {
            "idle": "idle",
            "working": "working…",
            "waiting": "waiting for feedback…",
            "stopped": "stopped",
        }
        self._agent_status.setText(label_map.get(state, state))

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        QTimer.singleShot(_STATUS_CLEAR_MS, lambda: self._status_label.setText(""))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_finished(self) -> None:
        """Remove listener when the dialog is closed/destroyed (WA_DeleteOnClose).

        Prevents a stale reference from calling Qt widget methods after deletion.
        In independent mode: detach (keeps supervisor running).
        In CLI mode: stop the child if it is still running.
        """
        self._chat.remove_listener(self._on_transcript_changed)
        if self._session is not None:
            if self._cli_mode and self._session.state in ("working", "waiting"):
                self._session.stop()
            else:
                self._session.detach()
