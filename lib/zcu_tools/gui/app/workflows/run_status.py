"""Workflow run status bar: badge, workflow, committed count and the paused banner."""

from dataclasses import dataclass
from typing import Literal

from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

type RunBadge = Literal[
    "ready",
    "pending",
    "running",
    "pausing",
    "paused",
    "stopping",
    "done",
    "aborted",
    "stopped",
    "failed",
]

# Live states are green, pause states amber, failures red, ends neutral.
_BADGE_COLORS: dict[RunBadge, str] = {
    "ready": "#6b7280",
    "pending": "#4b6584",
    "running": "#1f7a3a",
    "pausing": "#b26a00",
    "paused": "#b26a00",
    "stopping": "#9a5b13",
    "done": "#3b5b8a",
    "aborted": "#8a5a1c",
    "stopped": "#6b7280",
    "failed": "#b91c1c",
}
_PAUSED_NOTE = (
    "Paused. The uncommitted point is redone on Resume. You can Apply tunables first."
)
_PAUSED_STYLE = (
    "QLabel { background: #fdf3e1; border: 1px solid #ecc98a; border-radius: 3px;"
    " padding: 4px 8px; color: #3d2b06; }"
)


@dataclass(frozen=True)
class RunStatusLine:
    """One reading of the status bar.

    badge is "ready" before any run, "pending" before native acceptance, the
    native lifecycle once accepted, or "stopped"/"failed" for a run that ended
    before acceptance. workflow is the attached run's native name, or the
    selected one when no run is attached. detail is the plain text after it,
    for example "not started" or "3 committed · run 20260101-a". previous is
    True when the selection names another workflow than the attached run.
    """

    badge: RunBadge
    workflow: str
    detail: str
    previous: bool


class WorkflowRunStatus(QWidget):
    """Show one RunStatusLine above the paused banner.

    Object names: "workflow.run.status" (this widget), "workflow.run.previous"
    (the "Previous run:" prefix, shown only when the line says so),
    "workflow.run.badge", "workflow.run.workflow", "workflow.run.detail" and
    "workflow.run.paused_note" (the banner telling that Resume redoes the
    uncommitted point). Construct and call on the Qt GUI thread.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the bar reading "ready" with no workflow; the banner is hidden."""
        super().__init__(parent)
        self.setObjectName("workflow.run.status")
        self._previous = self._label("workflow.run.previous", "Previous run:")
        self._previous.setStyleSheet("QLabel { color: #6b7280; font-weight: 600; }")
        self._badge = self._label("workflow.run.badge", "")
        self._workflow = self._label("workflow.run.workflow", "")
        self._workflow.setStyleSheet("QLabel { font-weight: 600; }")
        self._detail = self._label("workflow.run.detail", "")
        self._paused = self._label("workflow.run.paused_note", _PAUSED_NOTE)
        self._paused.setWordWrap(True)
        self._paused.setStyleSheet(_PAUSED_STYLE)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for label in (self._previous, self._badge, self._workflow, self._detail):
            row.addWidget(label)
        row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self._paused)
        self._line: RunStatusLine | None = None
        self.show_line(RunStatusLine("ready", "", "", previous=False))
        self.show_paused_note(shown=False)

    def _label(self, name: str, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setObjectName(name)
        return label

    def show_line(self, line: RunStatusLine) -> None:
        """Show line's prefix, badge colour and text, workflow and detail."""
        # Polling repeats the same line; restyling only on change keeps it cheap.
        if line == self._line:
            return
        self._line = line
        self._previous.setVisible(line.previous)
        self._badge.setText(line.badge)
        self._badge.setStyleSheet(
            f"QLabel {{ background: {_BADGE_COLORS[line.badge]}; color: white;"
            " font-weight: 600; border-radius: 9px; padding: 1px 10px; }"
        )
        self._workflow.setText(line.workflow)
        self._detail.setText(line.detail)

    def show_paused_note(self, *, shown: bool) -> None:
        """Show or hide the banner that Resume redoes the uncommitted point."""
        self._paused.setVisible(shown)
