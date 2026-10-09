"""Concrete T1 tuning caller for the shared workflow submission policy."""

from collections.abc import Mapping

from pydantic import JsonValue
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
from zcu_tools.experiment.workflows import Actor, TunablesSnapshot
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.form import FormErrors, FormWritePath

from zcu_lab.workflows.calibration import Calibrated

from .panel import T1TunablesPanel
from .tunables import T1TunablesForm


class T1TuningPanel(QWidget):
    """Edit knobs for an already accepted native T1 run on the GUI thread.

    The composition root supplies the fixed Controller with an accepted binding.
    This panel presents that Host's unique FormState, without copying its input
    or capture. GUI and JSON Apply use the Controller's same submission entry.
    It does not Start, capture sources, own lifecycle or save input files.
    accepted_input is available on host for the app's persistence policy.
    panel is the existing T1 attachment; Engine owns numeric revisions.
    This transitional caller does not implement or check holder identity.
    """

    panel: T1TunablesPanel
    host: TunablesHost[T1TunablesForm, Calibrated]

    def __init__(
        self,
        controller: WorkflowController[T1TunablesForm, Calibrated],
        parent: QWidget | None = None,
    ) -> None:
        """Present controller's accepted Host on the QApplication/GUI thread.

        Missing/pending binding raises RuntimeError. The panel retains this run
        target; replacement makes its commands stale before any draft write.
        Source/editor errors propagate. parent controls Qt lifetime.
        """
        super().__init__(parent)
        binding = controller.binding
        if binding is None or binding.engine is None:
            raise RuntimeError("Tuning requires an accepted Controller binding")
        self._controller = controller
        self.host = binding
        self.panel = T1TunablesPanel(self.host.state, self.host.env, self)
        self._message = QLabel(self)
        self._message.setObjectName("workflow.tuning.message")
        apply_button = QPushButton("Apply", self)
        apply_button.setObjectName("workflow.tuning.apply")
        apply_button.clicked.connect(self._apply_clicked)
        discard_button = QPushButton("Discard", self)
        discard_button.setObjectName("workflow.tuning.discard")
        discard_button.clicked.connect(self.discard)
        layout = QVBoxLayout(self)
        layout.addWidget(self.panel)
        layout.addWidget(self._message)
        layout.addWidget(apply_button)
        layout.addWidget(discard_button)

    def apply(self) -> TunablesSnapshot | FormErrors:
        """Submit current complete input as a user, without a GUI version guard.

        Incomplete text/model errors return FormErrors without Engine edits.
        Native failures propagate without retry; draft/base are retained.
        """
        return self._controller.apply_tunables(
            self.host.run_id,
            actor=Actor("user", "gui"),
            local_errors=self.panel.editor.local_parse_errors(),
        )

    def apply_external(
        self,
        batch: Mapping[FormWritePath, JsonValue],
        *,
        actor: Actor,
    ) -> TunablesSnapshot | FormErrors:
        """Write GUI-RPC edits and submit complete current input.

        Parsed GUI draft participates. Invalid input stays in the draft and
        returns FormErrors; native failures propagate without retry. Success
        clears local unparseable text and shows the accepted input. No socket,
        direct MCP-to-Engine path or transitional holder check is introduced.
        """
        result = self._controller.apply_tunables(self.host.run_id, batch, actor=actor)
        if not isinstance(result, FormErrors):
            self.panel.editor.discard_local_text()
            self._message.setText(f"Applied revision {result.revision}")
        return result

    def discard(self) -> None:
        """Restore latest accepted input, including view-local invalid text."""
        self.host.discard()
        self.panel.editor.discard_local_text()
        self._message.clear()

    def _apply_clicked(self) -> None:
        try:
            result = self.apply()
        except ValueError as exc:
            self._message.setText(str(exc))
            return
        if isinstance(result, FormErrors):
            self._message.setText("; ".join(result.messages.values()))
        else:
            self._message.setText(f"Applied revision {result.revision}")

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Detach the embedded form subscription before normal close handling."""
        self.panel.editor.detach()
        super().closeEvent(a0)
