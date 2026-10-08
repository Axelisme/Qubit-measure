"""The renderer discards only its local unparseable text."""

import pytest
from pydantic import Field
from qtpy.QtWidgets import QApplication, QLineEdit
from zcu_tools.gui.form import FormEnv, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor


class NumericForm(FormModel):
    """One parsed numeric draft value independent of view-local text."""

    value: float = Field(default=1.0)


def test_discard_local_text_keeps_parsed_draft_and_restores_display() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    if not isinstance(app, QApplication):
        raise RuntimeError("Discard tests require a QApplication")
    env = FormEnv()
    state = FormState(NumericForm.default_input(env))
    editor = FormEditor(state, env)
    try:
        state.write({("value",): 4.0}, env)
        control = editor.findChild(QLineEdit, "form.input.value")
        assert isinstance(control, QLineEdit)
        control.setText("-")
        assert editor.local_parse_errors() is not None
        editor.discard_local_text()
        assert editor.local_parse_errors() is None
        assert state.input.data == {"value": 4.0}
        assert state.base.data == {"value": 1.0}
        assert float(control.text()) == 4.0
        editor.detach()
        with pytest.raises(RuntimeError, match="detach"):
            editor.discard_local_text()
    finally:
        editor.close()
        editor.deleteLater()
        app.processEvents()
