"""AgentLaunchDialog — open the system terminal running interactive ``claude``.

The final embedded-agent design (Round 1): a single non-modal dialog with two
buttons — **Resume last session** and **New session** — each spawns the OS's
default terminal running interactive claude against the loopback measure-gui
MCP server (see ``services.agent_launcher``). There is no transcript, input box,
or picker: the conversation happens in the user's own terminal.

The dialog is Qt-thin — all launch logic lives in the Qt-free
``agent_launcher`` module; the dialog only wires buttons to it and reports
success/failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.services import agent_launcher

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller

logger = logging.getLogger(__name__)


class AgentLaunchDialog(QDialog):
    """Two-button launcher for the external interactive ``claude`` terminal.

    ``Resume last session`` is disabled when no last session is persisted. Both
    buttons call ``agent_launcher.launch_agent_terminal`` and report the launched
    session id (or the failure) in the status label.
    """

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("Agent")

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Open the system terminal running an interactive claude agent "
                "wired to this GUI."
            )
        )

        button_row = QHBoxLayout()
        self._resume_btn = QPushButton("Resume last session")
        self._resume_btn.clicked.connect(self._on_resume)
        # Resume is only meaningful when a previous session id is on disk.
        self._resume_btn.setEnabled(agent_launcher.read_last_session_id() is not None)
        button_row.addWidget(self._resume_btn)

        self._new_btn = QPushButton("New session")
        self._new_btn.clicked.connect(self._on_new)
        button_row.addWidget(self._new_btn)
        layout.addLayout(button_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

    def _on_resume(self) -> None:
        self._launch(resume=True)

    def _on_new(self) -> None:
        self._launch(resume=False)

    def _launch(self, *, resume: bool) -> None:
        try:
            session_id = agent_launcher.launch_agent_terminal(
                self._ctrl.get_project_root(),
                resume=resume,
                state_context=self._ctrl.build_agent_state_context(),
            )
        except Exception as exc:  # noqa: BLE001 — surface any launch failure to UI
            logger.exception("AgentLaunchDialog: failed to launch agent terminal")
            self._status_label.setText(f"Failed to launch terminal: {exc}")
            return
        # A new session is now persisted as "last", so Resume becomes available.
        self._resume_btn.setEnabled(True)
        self._status_label.setText(f"Launched session {session_id} in terminal.")
