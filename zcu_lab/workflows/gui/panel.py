"""Concrete T1 tunables view for embedding in the workflow host."""

from collections.abc import Callable

from pydantic import BaseModel
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QVBoxLayout, QWidget
from zcu_tools.datafile.native_models import RunSnapshot
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor


class WorkflowFormPanel[F: FormModel](QWidget):
    """Render a shared app FormModel view of an attachment's unique draft.

    state is the supplied parsed input/base, not a copy. env is its fixed capture;
    editor owns local unparsed text only. No defaults, native submission, capture
    or lifecycle policy occurs. Closing detaches subscriptions, not the draft.
    T1 and overnight use this same view with independent model declarations.
    """

    def __init__(
        self, state: FormState[F], env: FormEnv, parent: QWidget | None = None
    ) -> None:
        """Embed the existing draft on the QApplication owner thread.

        Source/editor construction errors propagate; parent controls Qt lifetime.
        Hosts check editor.local_parse_errors before submitting parsed input.
        """
        super().__init__(parent)
        self.state = state
        self.env = env
        self.editor = FormEditor(state, env, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Detach editor subscriptions before normal Qt close handling."""
        self.editor.detach()
        super().closeEvent(a0)


class WorkflowPlanPanel(WorkflowFormPanel[FormModel]):
    """Create an attachment's plan draft and delegate native preview to its owner.

    state/editor/env follow WorkflowFormPanel. The paired converter materializes
    only supplied captured modules and historical snapshot; it never starts a run.
    """

    def __init__(
        self,
        model: type[FormModel],
        to_cfg: Callable[[FormModel, FormEnv, RunSnapshot], BaseModel | FormErrors],
        env: FormEnv,
        parent: QWidget | None = None,
    ) -> None:
        """Create partial defaults for model and its same-revision converter.

        Use on the QApplication owner. Required inputs remain absent until edited.
        Model/default/editor errors propagate without capture or native execution.
        """
        super().__init__(FormState(model.default_input(env)), env, parent)
        self._to_cfg = to_cfg

    def preview(self, snapshot: RunSnapshot) -> BaseModel | FormErrors:
        """Return a native plan or directed errors, without changing input/capture.

        snapshot is the host's historical entry/point/roles. Local unparsed text
        blocks preview. Native converter errors propagate; it owns plan validity.
        """
        errors = self.editor.local_parse_errors()
        if errors is not None:
            return errors
        result = self.state.result(self.env)
        if isinstance(result, FormErrors):
            return result
        return self._to_cfg(result, self.env, snapshot)
