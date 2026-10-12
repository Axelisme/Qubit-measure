"""Concrete workflow Window controls and their unchanged vertical layout."""

from dataclasses import dataclass

from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class WorkflowControls:
    """Actual Qt controls consumed by Window, without callbacks or run state.

    selector chooses the workflow; message displays run/input/view diagnostics.
    start/pause/resume/stop are the existing native run action buttons.
    export opens explicit merged/Labber export; samples appends scientific rows;
    setup opens shared Setup; aliases opens alias editing; prepare captures forms;
    reload attempts bounded source replacement. Their original order is retained.
    source_status displays current qubit availability, not admission authority.
    plot_stack and plot_placeholder are the existing FigureContainer attachment.
    layout is the original vertical layout used for replacing panels and tuning.
    Pause/resume/export/samples start disabled; all other buttons start enabled. Window
    sets reload availability, selector items and all signal/admission policy.
    """

    selector: QComboBox
    message: QLabel
    start: QPushButton
    pause: QPushButton
    resume: QPushButton
    stop: QPushButton
    export: QPushButton
    samples: QPushButton
    setup: QPushButton
    aliases: QPushButton
    prepare: QPushButton
    reload: QPushButton
    source_status: QLabel
    plot_stack: QStackedWidget
    plot_placeholder: QLabel
    layout: QVBoxLayout


def build_workflow_layout(
    parent: QWidget, plan_panel: QWidget, tunables_panel: QWidget
) -> WorkflowControls:
    """Build the existing controls/layout once on the Qt owner with QApplication.

    parent is their Window; plan_panel/tunables_panel are its existing selected
    panels. Preserves labels, objectNames, construction/tab order, initial flags
    and vertical order. Connects no signals and reads no state/source/hardware.
    Returned controls belong to parent; Window wires and applies admission policy.
    """
    selector = QComboBox(parent)
    selector.setObjectName("workflow.select")
    message = QLabel(parent)
    message.setObjectName("workflow.run.message")
    message.setWordWrap(True)
    start = QPushButton("Start", parent)
    start.setObjectName("workflow.run.start")
    pause = QPushButton("Pause", parent)
    pause.setObjectName("workflow.run.pause")
    pause.setEnabled(False)
    resume = QPushButton("Resume", parent)
    resume.setObjectName("workflow.run.resume")
    resume.setEnabled(False)
    stop = QPushButton("Stop", parent)
    stop.setObjectName("workflow.run.stop")
    export = QPushButton("Export merged/Labber", parent)
    export.setObjectName("workflow.run.export_fluxdep")
    export.setEnabled(False)
    samples = QPushButton("Export samples", parent)
    samples.setObjectName("workflow.run.export_samples")
    samples.setEnabled(False)
    plot_stack = QStackedWidget(parent)
    plot_placeholder = QLabel("Workflow plots", plot_stack)
    plot_stack.addWidget(plot_placeholder)
    layout = QVBoxLayout(parent)
    setup = QPushButton("Setup", parent)
    setup.setObjectName("workflow.source.setup")
    aliases = QPushButton("Aliases", parent)
    aliases.setObjectName("workflow.source.aliases")
    source_status = QLabel(parent)
    source_status.setObjectName("workflow.source.status")
    prepare = QPushButton("Prepare forms", parent)
    prepare.setObjectName("workflow.source.use")
    reload = QPushButton("Reload source", parent)
    reload.setObjectName("workflow.source.reload")
    for widget in (
        setup,
        aliases,
        source_status,
        prepare,
        reload,
        selector,
        plan_panel,
        tunables_panel,
        start,
        pause,
        resume,
        stop,
        export,
        samples,
        message,
        plot_stack,
    ):
        layout.addWidget(widget)
    return WorkflowControls(
        selector,
        message,
        start,
        pause,
        resume,
        stop,
        export,
        samples,
        setup,
        aliases,
        prepare,
        reload,
        source_status,
        plot_stack,
        plot_placeholder,
        layout,
    )
