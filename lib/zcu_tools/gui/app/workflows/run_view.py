"""Present the workflow Window's attached run on its fixed controls."""

from __future__ import annotations

from typing import Literal

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget

from zcu_tools.experiment.workflows import RunStatus
from zcu_tools.gui.form import FormModel

from .controller import WorkflowController
from .forms import WorkflowForms
from .layout import WorkflowControls
from .run_status import RunBadge, RunStatusLine
from .tuning_panel import WorkflowTuningPanel

_TUNABLES_TAB = 1
_POLL_MS = 50
_LIVE_LIFECYCLES = frozenset({"running", "pausing", "paused"})


class WorkflowRunView[C]:
    """Show the Controller's one attached run on the Window's controls.

    Status bar: badge, native workflow name, "n committed" and run_id; it reads
    "ready" with the selected workflow before any run, and "Previous run:"
    leads while the selection names another workflow than the attached run.
    From Start until terminal delivery the Start panels, selector and Start are
    disabled and the Plan tab shows its fixed-plan note. Once the run is
    accepted, its tuning panel replaces the Start knobs in the Tunables tab,
    titled "Tunables (live)"; it is enabled while running, pausing or paused.
    Paused delivery selects that tab; the banner shows while the run sits
    paused and hides once Resume is submitted. Terminal delivery shows the
    Start knobs again, reset to the run's latest accepted input, keeps the
    hidden tuning panel for remote calls, and puts a delivery error or native
    failure/abort reason in the diagnostics label. Pause is enabled while
    running, Resume while delivered paused, Stop while pending, running,
    pausing or paused.

    Reads the Controller and forms; never commands the run, captures sources or
    writes input files. The Window owns every other diagnostics message.
    Construct and call on the Qt owner thread.
    """

    def __init__(
        self,
        controls: WorkflowControls,
        controller: WorkflowController[FormModel, C],
        forms: WorkflowForms[C],
        parent: QWidget,
    ) -> None:
        """Present controller's run on controls; forms supplies the Start panels.

        parent owns the 50 ms status poll and each tuning panel. Shows the
        current state at once; polling runs from submitted() to terminal.
        """
        self._controls = controls
        self._controller = controller
        self._forms = forms
        self._parent = parent
        self._tuning: WorkflowTuningPanel[FormModel, C] | None = None
        self._knobs_from_run = False
        self._locked = False
        self._run_workflow: str | None = None
        self._ended_unaccepted: Literal["stopped", "failed"] | None = None
        self._poll = QTimer(parent)
        self._poll.setInterval(_POLL_MS)
        self._poll.timeout.connect(self.refresh)
        self.refresh()

    @property
    def tuning_panel(self) -> WorkflowTuningPanel[FormModel, C] | None:
        """Return the attached run's tuning panel, None before acceptance.

        It stays after terminal delivery, hidden, until a new Start, a panel
        replacement or close.
        """
        return self._tuning

    @property
    def knobs_from_run(self) -> bool:
        """Return whether the run's binding, not the Start panel, owns knobs.

        True from submitted() until the Start panels are replaced.
        """
        return self._knobs_from_run

    def release_tuning(self) -> None:
        """Close the previous run's tuning panel before a new Start submits."""
        if self._tuning is not None:
            self._tuning.close()
            self._tuning.deleteLater()
            self._tuning = None
        self._show_tunables(live=False)

    def submitted(self, workflow: str) -> None:
        """Lock the Start panels for the run just submitted for workflow.

        workflow is the run's native name. Clears the diagnostics label and
        starts polling.
        """
        self._run_workflow = workflow
        self._ended_unaccepted = None
        self._knobs_from_run = True
        self._locked = True
        controls = self._controls
        for widget in (
            self._forms.plan,
            self._forms.knobs,
            controls.selector,
            controls.start,
        ):
            widget.setEnabled(False)
        controls.message.clear()
        self._poll.start()
        self.refresh()

    def forms_replaced(self) -> None:
        """Show newly published Start panels; they do not present the run's knobs."""
        self._knobs_from_run = False
        self.release_tuning()
        self.refresh()

    def paused(self) -> None:
        """Show paused delivery: select the live Tunables tab, keep polling."""
        self.refresh()
        self._controls.forms.setCurrentIndex(_TUNABLES_TAB)
        self._poll.start()

    def ended(self, status: RunStatus | None, error: Exception | None) -> None:
        """Show terminal delivery and unlock Start.

        status is the native terminal status, None when the run ended before
        acceptance (stopped without error, failed with one). error is the
        producer's original error, if any.
        """
        self._poll.stop()
        if status is None:
            self._ended_unaccepted = "stopped" if error is None else "failed"
        self._locked = False
        binding = self._controller.binding
        knobs = self._forms.knobs
        if binding is not None and self._knobs_from_run:
            knobs.state.reset(binding.accepted_input)
            knobs.editor.discard_local_text()
        controls = self._controls
        for widget in (self._forms.plan, knobs, controls.selector, controls.start):
            widget.setEnabled(True)
        if error is not None:
            controls.message.setText(str(error))
        elif (
            status is not None
            and status.lifecycle in ("failed", "aborted")
            and status.reason is not None
        ):
            controls.message.setText(status.reason)
        self.refresh()

    def close(self) -> None:
        """Stop polling and close the tuning panel when the Window closes."""
        self._poll.stop()
        if self._tuning is not None:
            self._tuning.close()

    def refresh(self) -> None:
        """Show the current run state; build the tuning panel once accepted."""
        controller = self._controller
        status = controller.status()
        lifecycle = None if status is None else status.lifecycle
        binding = controller.binding
        if (
            self._knobs_from_run
            and self._tuning is None
            and binding is not None
            and binding.engine is not None
        ):
            self._tuning = WorkflowTuningPanel(
                controller,
                self._parent,
                panel_factory=self._forms.attachment.knobs_panel,
            )
            page = self._controls.tunables_page
            page.insertWidget(page.indexOf(self._forms.knobs) + 1, self._tuning)
        live = lifecycle in _LIVE_LIFECYCLES
        if self._tuning is not None:
            self._tuning.setEnabled(live)
        self._show_tunables(live=self._locked and self._tuning is not None)
        pending = controller.engine is None and controller.has_pending_segment
        controls = self._controls
        controls.pause.setEnabled(
            controller.operation_id is not None and lifecycle == "running"
        )
        delivered_paused = lifecycle == "paused" and not controller.has_pending_segment
        controls.resume.setEnabled(delivered_paused)
        controls.stop.setEnabled(pending or live)
        controls.status.show_line(self._line(status))
        controls.status.show_paused_note(shown=delivered_paused)

    def _line(self, status: RunStatus | None) -> RunStatusLine:
        selected = self._forms.attachment.name
        binding = self._controller.binding
        workflow = self._run_workflow
        # submitted() names the workflow of every binding Window creates.
        if binding is None or workflow is None:
            return RunStatusLine("ready", selected, "not started", previous=False)
        run = f"run {binding.run_id}"
        badge: RunBadge
        if status is not None:
            badge = status.lifecycle
            detail = f"{status.committed_seq} committed · {run}"
        elif self._ended_unaccepted == "stopped":
            badge, detail = "stopped", f"stopped before startup · {run}"
        elif self._ended_unaccepted == "failed":
            badge, detail = "failed", f"startup failed · {run}"
        else:
            badge, detail = "pending", f"startup pending · {run}"
        return RunStatusLine(badge, workflow, detail, previous=workflow != selected)

    def _show_tunables(self, *, live: bool) -> None:
        """Show the live tuning panel or the Start knobs in the Tunables tab."""
        controls = self._controls
        self._forms.knobs.setVisible(not live)
        if self._tuning is not None:
            self._tuning.setVisible(live)
        title = "Tunables (live)" if live else "Tunables"
        controls.forms.setTabText(_TUNABLES_TAB, title)
        controls.tunables_note.setVisible(live)
        controls.plan_note.setVisible(self._locked)
