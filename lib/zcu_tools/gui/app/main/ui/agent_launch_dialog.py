"""AgentLaunchDialog — open the system terminal running interactive ``claude``.

The embedded-agent design (Round 1): a non-modal dialog that lists previously
launched sessions so the user can pick one to resume, plus a **New session**
button for a fresh start. Each action spawns the OS's default terminal running
interactive claude against the loopback measure-gui MCP server (see
``services.agent_launcher``). There is no transcript, input box, or live preview:
the conversation happens in the user's own terminal.

The dialog is Qt-thin — all launch logic lives in the Qt-free
``agent_launcher`` module; the dialog only wires the list and buttons to it and
reports success/failure via the status label.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.services import agent_launcher

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller

logger = logging.getLogger(__name__)

# Sentinel stored in each QListWidgetItem's UserRole data to identify which
# session_id the item represents.
_SESSION_ID_ROLE = Qt.ItemDataRole.UserRole


def _format_relative_time(epoch: float) -> str:
    """Return a human-readable relative time string for a Unix epoch timestamp.

    Examples: "just now", "5 min ago", "3 h ago", "2 d ago".
    """
    delta = time.time() - epoch
    if delta < 60:
        return "just now"
    if delta < 3600:
        mins = int(delta / 60)
        return f"{mins} min ago"
    if delta < 86400:
        hrs = int(delta / 3600)
        return f"{hrs} h ago"
    days = int(delta / 86400)
    return f"{days} d ago"


class AgentLaunchDialog(QDialog):
    """Session-list launcher for the external interactive ``claude`` terminal.

    Displays a scrollable list of previously launched sessions (label + relative
    time + short id). The user selects a row to enable **Resume selected**. A
    **New session** button always launches a fresh session. A **Refresh** button
    reloads the list from disk.
    """

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("Agent")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Open the system terminal running an interactive claude agent "
                "wired to this GUI."
            )
        )

        layout.addWidget(QLabel("Previous sessions:"))

        # Session list — each item stores the session_id in UserRole data.
        self._session_list = QListWidget()
        self._session_list.setAlternatingRowColors(True)
        self._session_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._session_list)

        button_row = QHBoxLayout()

        self._resume_btn = QPushButton("Resume selected")
        # Enabled only when a list item is selected.
        self._resume_btn.setEnabled(False)
        self._resume_btn.clicked.connect(self._on_resume)
        button_row.addWidget(self._resume_btn)

        self._new_btn = QPushButton("New session")
        self._new_btn.clicked.connect(self._on_new)
        button_row.addWidget(self._new_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._reload_sessions)
        button_row.addWidget(self._refresh_btn)

        layout.addLayout(button_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Populate list on open.
        self._reload_sessions()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reload_sessions(self) -> None:
        """Refresh the session list from disk."""
        self._session_list.clear()
        try:
            sessions = agent_launcher.list_resumable_sessions(
                self._ctrl.get_project_root()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AgentLaunchDialog: failed to load sessions: %s", exc)
            sessions = []

        for session in sessions:
            rel_time = _format_relative_time(session.last_active)
            short_id = session.session_id[:8]
            display = f"{session.label}  [{rel_time} · {short_id}]"
            item = QListWidgetItem(display)
            item.setData(_SESSION_ID_ROLE, session.session_id)
            self._session_list.addItem(item)

        # Select the first (most recent) item automatically when available.
        if self._session_list.count() > 0:
            self._session_list.setCurrentRow(0)

        self._update_resume_btn()

    def _selected_session_id(self) -> str | None:
        """Return the session_id of the currently selected list item, or None."""
        items = self._session_list.selectedItems()
        if not items:
            return None
        data = items[0].data(_SESSION_ID_ROLE)
        return data if isinstance(data, str) else None

    def _update_resume_btn(self) -> None:
        self._resume_btn.setEnabled(self._selected_session_id() is not None)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        self._update_resume_btn()

    def _on_resume(self) -> None:
        session_id = self._selected_session_id()
        if session_id is None:
            # Should not happen (button is disabled when nothing selected),
            # but guard defensively.
            self._status_label.setText("No session selected.")
            return
        self._launch(resume_session_id=session_id)

    def _on_new(self) -> None:
        self._launch(resume_session_id=None)

    def _launch(self, *, resume_session_id: str | None) -> None:
        try:
            session_id = agent_launcher.launch_agent_terminal(
                self._ctrl.get_project_root(),
                resume_session_id=resume_session_id,
                state_context=self._ctrl.build_agent_state_context(),
            )
        except Exception as exc:  # noqa: BLE001 — surface any launch failure to UI
            logger.exception("AgentLaunchDialog: failed to launch agent terminal")
            self._status_label.setText(f"Failed to launch terminal: {exc}")
            return
        # Terminal launched (Resume or New) — close the dialog. The conversation
        # now lives in the user's terminal; reopening the dialog re-reads the
        # session list (so the freshly launched session shows up next time).
        self._status_label.setText(f"Launched session {session_id} in terminal.")
        self.accept()
