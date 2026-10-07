"""Concrete T1 tunables view for embedding in the workflow host."""

from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QVBoxLayout, QWidget
from zcu_tools.gui.form import FormEnv, FormState
from zcu_tools.gui.widgets.form import FormEditor

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
