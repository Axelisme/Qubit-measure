"""Concrete T1 tuning caller for the shared workflow submission policy."""

from collections.abc import Mapping

from pydantic import JsonValue
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
from zcu_tools.experiment.workflows import Actor, Engine, TunablesSnapshot
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormWritePath

from zcu_lab.workflows.calibration import Calibrated

from .panel import T1TunablesPanel
from .tunables import T1TunablesForm


class T1TuningPanel(QWidget):
    """Edit knobs for an already accepted native T1 run on the GUI thread.

    The composition root supplies engine/run_id, the fixed form env and exact
    accepted input used for Start. This panel does not Start, capture sources,
    own worker/lease lifecycle or save input files. GUI and JSON callers share
    host. accepted_input is available on host for the app's persistence policy.
    panel is the existing T1 attachment; Engine owns numeric revisions.
    This transitional caller does not implement or check holder identity.
    """

    panel: T1TunablesPanel
    host: TunablesHost[T1TunablesForm, Calibrated]

    def __init__(
        self,
        engine: Engine[Calibrated],
        run_id: str,
        env: FormEnv,
        accepted_input: FormInput[T1TunablesForm],
        parent: QWidget | None = None,
    ) -> None:
        """Embed the T1 attachment bound to an existing Engine run.

        Requires QApplication/GUI thread. Wrong run IDs and form classes or
        source/editor errors propagate. parent controls Qt lifetime.
        """
        super().__init__(parent)
        self.panel = T1TunablesPanel(env, self)
        self.panel.state.reset(accepted_input)
        self.host = TunablesHost(self.panel.state, env, engine, run_id)
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
        return self.host.apply(
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
        result = self.host.apply_external(batch, actor=actor)
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
