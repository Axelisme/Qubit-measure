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
    """Present a host-owned T1 draft on the GUI thread.

    env is the constructor's fixed capture, not a live source.
    state is the supplied FormState[T1TunablesForm], not a copy or fresh draft.
    editor is its FormEditor evaluated in the supplied fixed env. Hosts use
    these public handles for validation, changes and reset, and check
    editor.local_parse_errors before submission. This panel does not create
    defaults, submit, capture sources or own native lifecycle.
    """

    env: FormEnv
    state: FormState[T1TunablesForm]
    editor: FormEditor[T1TunablesForm]

    def __init__(
        self,
        state: FormState[T1TunablesForm],
        env: FormEnv,
        parent: QWidget | None = None,
    ) -> None:
        """Embed an editor for state without resetting its input or base.

        state is the host's T1 draft. env is its fixed evaluation capture,
        not live sources. Editor construction failures propagate. parent
        controls Qt lifetime; closing detaches the editor but preserves state.
        Requires GUI-thread use and a QApplication.
        """
        super().__init__(parent)
        self.env = env
        self.state = state
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

    env is the constructor's fixed capture, not a live source.
    state owns the editable T1PlanForm input. editor presents it and retains
    local parse errors. Both use the fixed host env supplied at construction.
    Host chooses the historical snapshot and owns Start/read-only policy.
    """

    env: FormEnv
    state: FormState[T1PlanForm]
    editor: FormEditor[T1PlanForm]

    def __init__(self, env: FormEnv, parent: QWidget | None = None) -> None:
        """Create a partial draft on the GUI thread with an existing QApplication.

        Required module selections remain absent until host/user input supplies
        them. env is borrowed unchanged. Source/editor failures propagate and
        parent controls Qt lifetime. No Engine or live resource is accessed.
        """
        super().__init__(parent)
        self.env = env
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
        result = self.state.result(self.env)
        if isinstance(result, FormErrors):
            return result
        return result.to_cfg(self.env, snapshot)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Detach the editor subscriptions before normal Qt close handling."""
        self.editor.detach()
        super().closeEvent(a0)
