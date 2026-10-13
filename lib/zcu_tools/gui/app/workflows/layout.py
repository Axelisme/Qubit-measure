"""Concrete workflow Window controls: a session strip above two resizable columns."""

from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._palette import PRIMARY_COLOR
from .run_status import WorkflowRunStatus

# Start is the primary action; Stop reads as destructive. Disabled states are
# visibly grey so an unavailable action does not look clickable.
_START_STYLE = (
    f"QPushButton {{ background-color: {PRIMARY_COLOR}; color: white;"
    " font-weight: 600;"
    " border: 1px solid #205aa9; border-radius: 4px; padding: 5px 12px; }"
    "QPushButton:hover:!disabled { background-color: #2f76dc; }"
    "QPushButton:disabled { background-color: #c3ccd8; color: #eef1f5;"
    " border-color: #b3bdca; }"
)
_STOP_STYLE = (
    "QPushButton { color: #b91c1c; font-weight: 600; }"
    "QPushButton:disabled { color: #a3a8b0; font-weight: normal; }"
)
_NOTE_STYLE = (
    "QLabel { background: #eef4fc; border: 1px solid #c9daf2; border-radius: 3px;"
    " padding: 4px 8px; color: #1f2d44; }"
)
_PLAN_NOTE = "Plan is fixed for this run. Edit it after the run ends."
_TUNABLES_NOTE = "Live tunables: Apply changes the points not yet committed."
_FORM_COLUMN_WIDTH = 560
_FIGURE_COLUMN_WIDTH = 1040


@dataclass(frozen=True)
class WorkflowControls:
    """Actual Qt controls consumed by Window, without callbacks or run state.

    Top row: the session strip, then setup (opens shared Setup), aliases (opens
    alias editing), devices (opens shared device management), prepare
    (captures forms) and reload (bounded source replacement). Below it a
    splitter holds two user-resizable columns.
    Left: selector chooses the workflow above forms, whose tab 0 is Plan and
    tab 1 is Tunables; tunables_tab is the Tunables tab's page widget, so
    forms.indexOf(tunables_tab) locates that tab without assuming its index;
    plan_page and tunables_page are the layouts inside each
    tab's scroll area that hold the current Start panels (and the live tuning
    panel); plan_note and tunables_note sit above each scroll area, hidden,
    reading that the plan is fixed and that live Apply changes uncommitted
    points; start/pause/resume/stop are the native run buttons fixed under the
    tabs. Right: status is the run status bar with its paused banner; message
    shows input/command/run diagnostics under it, above the caller's figures.
    Pause/resume start disabled; all other buttons start enabled. Window
    sets prepare/reload availability, selector items and all signal/admission
    policy.
    """

    selector: QComboBox
    status: WorkflowRunStatus
    message: QLabel
    start: QPushButton
    pause: QPushButton
    resume: QPushButton
    stop: QPushButton
    setup: QPushButton
    aliases: QPushButton
    devices: QPushButton
    prepare: QPushButton
    reload: QPushButton
    forms: QTabWidget
    plan_page: QVBoxLayout
    tunables_page: QVBoxLayout
    tunables_tab: QWidget
    plan_note: QLabel
    tunables_note: QLabel


def _button(text: str, name: str, parent: QWidget) -> QPushButton:
    button = QPushButton(text, parent)
    button.setObjectName(name)
    return button


def _scroll_tab(
    forms: QTabWidget, title: str, panel: QWidget, note_text: str
) -> tuple[QWidget, QVBoxLayout, QLabel]:
    """Add a tab: a hidden note above content that scrolls.

    Return the tab's page widget, the layout holding panel and the note.
    """
    tab = QWidget(forms)
    tab_layout = QVBoxLayout(tab)
    tab_layout.setContentsMargins(0, 0, 0, 0)
    name = f"workflow.forms.{title.lower()}"
    note = QLabel(note_text, tab)
    note.setObjectName(f"{name}.note")
    note.setWordWrap(True)
    note.setStyleSheet(_NOTE_STYLE)
    note.hide()
    tab_layout.addWidget(note)
    scroll = QScrollArea(tab)
    scroll.setObjectName(name)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    page = QWidget(scroll)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    layout.addStretch(1)
    scroll.setWidget(page)
    tab_layout.addWidget(scroll, 1)
    forms.addTab(tab, title)
    return tab, layout, note


def _run_buttons(
    parent: QWidget,
) -> tuple[QPushButton, QPushButton, QPushButton, QPushButton]:
    """Return Start (primary), Pause and Resume (disabled) and Stop (red text)."""
    start = _button("Start", "workflow.run.start", parent)
    start.setStyleSheet(_START_STYLE)
    pause = _button("Pause", "workflow.run.pause", parent)
    pause.setEnabled(False)
    resume = _button("Resume", "workflow.run.resume", parent)
    resume.setEnabled(False)
    stop = _button("Stop", "workflow.run.stop", parent)
    stop.setStyleSheet(_STOP_STYLE)
    return start, pause, resume, stop


def _figure_column(
    splitter: QSplitter, figures: QWidget
) -> tuple[QWidget, WorkflowRunStatus, QLabel]:
    """Return the right column: run status, then diagnostics, then figures."""
    column = QWidget(splitter)
    layout = QVBoxLayout(column)
    layout.setContentsMargins(4, 0, 0, 0)
    status = WorkflowRunStatus(column)
    layout.addWidget(status)
    message = QLabel(column)
    message.setObjectName("workflow.run.message")
    message.setWordWrap(True)
    layout.addWidget(message)
    layout.addWidget(figures, 1)
    return column, status, message


def build_workflow_layout(
    parent: QWidget,
    session: QWidget,
    figures: QWidget,
    plan_panel: QWidget,
    tunables_panel: QWidget,
) -> WorkflowControls:
    """Build the controls and their layout once on the Qt owner with QApplication.

    parent is their Window. session is its already built strip of read-only
    session chips; figures is its already built figure area, placed under the
    run status and diagnostics. plan_panel/tunables_panel are the selected Start panels,
    placed in the Plan and Tunables tabs. Keeps every existing objectName and initial
    flag. Connects no signals and reads no state/source/hardware.
    Returned controls belong to parent; Window wires and applies admission policy.
    """
    root = QVBoxLayout(parent)
    strip = QHBoxLayout()
    strip.addWidget(session)
    strip.addStretch(1)
    setup = _button("Setup…", "workflow.source.setup", parent)
    aliases = _button("Aliases…", "workflow.source.aliases", parent)
    devices = _button("Devices…", "workflow.source.devices", parent)
    prepare = _button("Prepare forms", "workflow.source.use", parent)
    reload = _button("Reload source", "workflow.source.reload", parent)
    for button in (setup, aliases, devices, prepare, reload):
        strip.addWidget(button)
    root.addLayout(strip)

    splitter = QSplitter(Qt.Orientation.Horizontal, parent)
    splitter.setObjectName("workflow.split")
    splitter.setChildrenCollapsible(False)
    left = QWidget(splitter)
    left_layout = QVBoxLayout(left)
    left_layout.setContentsMargins(0, 0, 4, 0)
    choose = QHBoxLayout()
    choose.addWidget(QLabel("Workflow", left))
    selector = QComboBox(left)
    selector.setObjectName("workflow.select")
    choose.addWidget(selector, 1)
    left_layout.addLayout(choose)
    forms = QTabWidget(left)
    forms.setObjectName("workflow.forms")
    # Only the Tunables tab is located later (paused delivery selects it).
    _, plan_page, plan_note = _scroll_tab(forms, "Plan", plan_panel, _PLAN_NOTE)
    tunables_tab, tunables_page, tunables_note = _scroll_tab(
        forms, "Tunables", tunables_panel, _TUNABLES_NOTE
    )
    left_layout.addWidget(forms, 1)
    run_row = QHBoxLayout()
    start, pause, resume, stop = _run_buttons(left)
    for button in (start, pause, resume, stop):
        run_row.addWidget(button)
    left_layout.addLayout(run_row)
    right, status, message = _figure_column(splitter, figures)

    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([_FORM_COLUMN_WIDTH, _FIGURE_COLUMN_WIDTH])
    root.addWidget(splitter, 1)
    return WorkflowControls(
        selector,
        status,
        message,
        start,
        pause,
        resume,
        stop,
        setup,
        aliases,
        devices,
        prepare,
        reload,
        forms,
        plan_page,
        tunables_page,
        tunables_tab,
        plan_note,
        tunables_note,
    )
