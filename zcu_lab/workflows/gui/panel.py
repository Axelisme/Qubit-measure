"""Concrete T1 tunables view for embedding in the workflow host."""

from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QVBoxLayout, QWidget
from zcu_tools.datafile.native_models import RunSnapshot
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.gui.widgets.form import FormEditor

from zcu_lab.workflows.t1 import T1Plan

from .plan import T1PlanForm
from .tunables import T1TunablesForm


class T1TunablesPanel(QWidget):
    """Create a T1 draft and its editor from host-captured env on the GUI thread.

    state is the panel-owned FormState[T1TunablesForm] with input defaults from
    env. editor is its FormEditor. Hosts use these public handles for validation,
    changes and reset, and check editor.local_parse_errors before submission.
    The panel neither submits nor captures sources, and has no native preview,
    Start, Apply or Discard action. F5 embeds this panel directly.
    """

    state: FormState[T1TunablesForm]
    editor: FormEditor[T1TunablesForm]

    def __init__(self, env: FormEnv, parent: QWidget | None = None) -> None:
        """Create state with T1 defaults and embed its editor in this QWidget.

        env is supplied by the host; it is not refreshed or copied from live
        sources. Default/source/editor construction failures propagate. parent
        controls the Qt lifetime. Requires GUI-thread use and a QApplication.
        """
        super().__init__(parent)
        self.state = FormState(T1TunablesForm.default_input(env))
        self.editor = FormEditor(self.state, env, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Detach before Qt handles close event a0, preserving the draft.

        Qt supplies the event; its binding permits None in the override type.
        Event acceptance remains the QWidget default.
        """
        self.editor.detach()
        super().closeEvent(a0)


class T1PlanPanel(QWidget):
    """Edit native T1 plan input and preview without submitting or capturing.

    state owns the editable T1PlanForm input. editor presents it and retains
    local parse errors. Both use the fixed host env supplied at construction.
    Host chooses the historical snapshot and owns Start/read-only policy.
    """

    state: FormState[T1PlanForm]
    editor: FormEditor[T1PlanForm]

    def __init__(self, env: FormEnv, parent: QWidget | None = None) -> None:
        """Create a partial draft on the GUI thread with an existing QApplication.

        Required module selections remain absent until host/user input supplies
        them. env is borrowed unchanged. Source/editor failures propagate and
        parent controls Qt lifetime. No Engine or live resource is accessed.
        """
        super().__init__(parent)
        self._env = env
        self.state = FormState(T1PlanForm.default_input(env))
        self.editor = FormEditor(self.state, env, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)

    def preview(self, snapshot: RunSnapshot) -> T1Plan | FormErrors:
        """Return a native plan or directed parse/validation errors without Start.

        snapshot is historical entry/point/roles supplied by the host. Local
        incomplete text blocks preview even when the retained draft is valid.
        Native cfg failures propagate; input and source capture stay unchanged.
        """
        errors = self.editor.local_parse_errors()
        if errors is not None:
            return errors
        result = self.state.result(self._env)
        if isinstance(result, FormErrors):
            return result
        return result.to_cfg(self._env, snapshot)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Detach the editor subscriptions before normal Qt close handling."""
        self.editor.detach()
        super().closeEvent(a0)
