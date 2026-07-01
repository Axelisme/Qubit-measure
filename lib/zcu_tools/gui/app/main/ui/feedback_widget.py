"""FeedbackPanel — docked, collapsible user→agent feedback panel.

A ``_CollapsibleSection`` (the same ▼/▶ header idiom as the in-tab
Analysis/Writeback sections) that holds a one-line message input plus
"Send" / "Send & Stop" buttons. FeedbackDockController mounts it directly
below the figure of the target tab (running tab if any, else the active tab)
and unmounts it again, so it docks under the plot rather than floating over it.

Visibility (the C3 gate, ADR-0025) is owned by FeedbackDockController: the
panel is mounted only while at least one live operation is in progress AND at
least one MCP control client is connected. MainWindow keeps the public
refresh_feedback_widget() facade for bus handlers and
RemoteControlAdapter._on_client_count_changed() so both triggers converge on
the same idempotent decision.

The panel is app-level (single foreground operation => single feedback) and
lives on the Qt main thread; send_feedback -> operation channel path is
unchanged (ADR-0025).

Stop-gating: 'Send & Stop' is enabled only when the active operation has a
cancel hook registered (ADR-0025 §Stop-gating). Gating is refreshed by
FeedbackDockController each time the op count or op type changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from .fields import _CollapsibleSection

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller

logger = logging.getLogger(__name__)


class FeedbackPanel(_CollapsibleSection):
    """Collapsible 'Send to agent' panel docked below the figure.

    Public API (called by the dock controller):
    - refresh_gating(): re-read can_cancel_active_operation() and
      enable/disable 'Send & Stop' accordingly.
    - clear_input(): wipe the text field (called on unmount).
    """

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        # Default EXPANDED (collapsed=False); the user can collapse via header.
        super().__init__(
            "Send to agent", collapsible=True, collapsed=False, parent=parent
        )
        self._ctrl = ctrl

        self._input = QLineEdit()
        self._input.setPlaceholderText("Message…")
        self._input.textChanged.connect(self._on_text_changed)
        self.body_layout.addWidget(self._input)

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

        self.body_layout.addLayout(btn_row)

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
        """Clear the text field (called when the panel is unmounted)."""
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
        logger.debug("FeedbackPanel: send nudge %r", text)
        self._ctrl.send_feedback(text, stop=False)
        self._input.clear()

    def _on_send_stop_clicked(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        logger.debug("FeedbackPanel: send+stop %r", text)
        self._ctrl.send_feedback(text, stop=True)
        self._input.clear()
