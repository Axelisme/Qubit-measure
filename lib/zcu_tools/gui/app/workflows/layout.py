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
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Start is the primary action; Stop reads as destructive. Disabled states are
# visibly grey so an unavailable action does not look clickable.
_START_STYLE = (
    "QPushButton { background-color: #286ac7; color: white; font-weight: 600;"
    " border: 1px solid #205aa9; border-radius: 4px; padding: 5px 12px; }"
    "QPushButton:hover:!disabled { background-color: #2f76dc; }"
    "QPushButton:disabled { background-color: #c3ccd8; color: #eef1f5;"
    " border-color: #b3bdca; }"
)
_STOP_STYLE = (
    "QPushButton { color: #b91c1c; font-weight: 600; }"
    "QPushButton:disabled { color: #a3a8b0; font-weight: normal; }"
)
_FORM_COLUMN_WIDTH = 560
_FIGURE_COLUMN_WIDTH = 1040


@dataclass(frozen=True)
class WorkflowControls:
    """Actual Qt controls consumed by Window, without callbacks or run state.

    Top row: the session strip, then setup (opens shared Setup), aliases (opens
    alias editing), prepare (captures forms) and reload (bounded source
    replacement). Below it a splitter holds two user-resizable columns.
    Left: selector chooses the workflow above the Plan and Tunables tabs;
    plan_page and tunables_page are the layouts inside each tab's scroll area
    that hold the current Start panels; start/pause/resume/stop are the native
    run buttons fixed under the tabs. Right: message shows run/input/view
    diagnostics above plot_stack; plot_stack and plot_placeholder are the
    existing FigureContainer attachment.
    Pause/resume start disabled; all other buttons start enabled. Window
    sets reload availability, selector items and all signal/admission policy.
    """

    selector: QComboBox
    message: QLabel
    start: QPushButton
    pause: QPushButton
    resume: QPushButton
    stop: QPushButton
    setup: QPushButton
    aliases: QPushButton
    prepare: QPushButton
    reload: QPushButton
    plan_page: QVBoxLayout
    tunables_page: QVBoxLayout
    plot_stack: QStackedWidget
    plot_placeholder: QLabel


def _button(text: str, name: str, parent: QWidget) -> QPushButton:
    button = QPushButton(text, parent)
    button.setObjectName(name)
    return button


def _scroll_tab(forms: QTabWidget, title: str, panel: QWidget) -> QVBoxLayout:
    """Add a tab whose content scrolls; return the layout holding panel."""
    scroll = QScrollArea(forms)
    scroll.setObjectName(f"workflow.forms.{title.lower()}")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    page = QWidget(scroll)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    layout.addStretch(1)
    scroll.setWidget(page)
    forms.addTab(scroll, title)
    return layout


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
    splitter: QSplitter,
) -> tuple[QWidget, QLabel, QStackedWidget, QLabel]:
    """Return the right column, its run message, plot stack and stack placeholder."""
    column = QWidget(splitter)
    layout = QVBoxLayout(column)
    layout.setContentsMargins(4, 0, 0, 0)
    message = QLabel(column)
    message.setObjectName("workflow.run.message")
    message.setWordWrap(True)
    layout.addWidget(message)
    plot_stack = QStackedWidget(column)
    plot_stack.setObjectName("workflow.figures")
    placeholder = QLabel("Workflow plots", plot_stack)
    placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
    plot_stack.addWidget(placeholder)
    layout.addWidget(plot_stack, 1)
    return column, message, plot_stack, placeholder


def build_workflow_layout(
    parent: QWidget,
    session: QWidget,
    plan_panel: QWidget,
    tunables_panel: QWidget,
) -> WorkflowControls:
    """Build the controls and their layout once on the Qt owner with QApplication.

    parent is their Window. session is its already built strip of read-only
    session chips. plan_panel/tunables_panel are the selected Start panels,
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
    prepare = _button("Prepare forms", "workflow.source.use", parent)
    reload = _button("Reload source", "workflow.source.reload", parent)
    for button in (setup, aliases, prepare, reload):
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
    plan_page = _scroll_tab(forms, "Plan", plan_panel)
    tunables_page = _scroll_tab(forms, "Tunables", tunables_panel)
    left_layout.addWidget(forms, 1)
    run_row = QHBoxLayout()
    start, pause, resume, stop = _run_buttons(left)
    for button in (start, pause, resume, stop):
        run_row.addWidget(button)
    left_layout.addLayout(run_row)
    right, message, plot_stack, plot_placeholder = _figure_column(splitter)

    splitter.addWidget(left)
    splitter.addWidget(right)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([_FORM_COLUMN_WIDTH, _FIGURE_COLUMN_WIDTH])
    root.addWidget(splitter, 1)
    return WorkflowControls(
        selector,
        message,
        start,
        pause,
        resume,
        stop,
        setup,
        aliases,
        prepare,
        reload,
        plan_page,
        tunables_page,
        plot_stack,
        plot_placeholder,
    )
