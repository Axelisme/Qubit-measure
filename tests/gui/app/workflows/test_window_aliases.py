"""The Window's Aliases entry edits the global source, not the captured forms.

The shared AliasDialog owns Apply/Cancel/invalid semantics (tests/gui/session/ui).
Here the host contract is: the button opens that dialog on this session, the
session commits the alias, the current panels, drafts, fixed capture and local
text stay, and Prepare forms uses the new alias.
"""

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QLineEdit,
    QPushButton,
    QTableWidget,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.ui.alias_dialog import AliasDialog

from ._window_support import (
    QUBIT_KIND,
    WideKnobsForm,
    WidePlanForm,
    WindowSession,
    bind_point,
)


def prepared_with_local_text(session: WindowSession) -> QLineEdit:
    """Prepare forms for Q1, edit both drafts, and leave unparsed knobs text."""
    window = session.window
    bind_point(session.harness, session.root)
    session.harness.services.context_control.add_component("Q2", QUBIT_KIND, {})
    window.prepare_forms()
    env = window.plan_panel.env
    assert env.parameters is not None and env.parameters.resolve_path("$qubit") == "Q1"
    window.plan_panel.state.write({("count",): 9}, env)
    window.tunables_panel.state.write({("value",): 4}, env)
    edit = window.tunables_panel.findChild(QLineEdit, "form.input.value")
    assert edit is not None
    edit.setText("1+")
    edit.editingFinished.emit()
    return edit


def edit_qubit_alias(
    window: WorkflowWindow[None], qapp: QApplication, value_text: str, *, apply: bool
) -> None:
    """Open Aliases from the Window, type the qubit value, optionally Apply, Cancel."""
    entry = window.findChild(QPushButton, "workflow.source.aliases")
    assert entry is not None
    entry.click()
    dialog = next((d for d in window.findChildren(AliasDialog) if d.isVisible()), None)
    assert dialog is not None
    table = dialog.findChild(QTableWidget)
    buttons = dialog.findChild(QDialogButtonBox)
    assert table is not None and buttons is not None
    value = table.item(0, 1)
    assert value is not None and value.text() == "Q1"
    value.setText(value_text)
    if apply:
        button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        assert button is not None
        button.click()
        assert dialog.isVisible()
    cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
    assert cancel is not None
    cancel.click()
    qapp.processEvents()


@pytest.mark.parametrize("outcome", ["apply", "cancel", "invalid"])
def test_alias_entry_edits_same_session_without_replacing_captured_forms(
    window_session: WindowSession, qapp: QApplication, outcome: str
) -> None:
    window = window_session.window
    edit = prepared_with_local_text(window_session)
    plan, knobs = window.plan_panel, window.tunables_panel
    env = plan.env
    before = plan.state.input.data, knobs.state.input.data
    bases = plan.state.base.data, knobs.state.base.data

    edit_qubit_alias(
        window,
        qapp,
        "$qubit" if outcome == "invalid" else "Q2",
        apply=outcome != "cancel",
    )
    expected = "Q2" if outcome == "apply" else "Q1"
    control = window_session.harness.services.context_control
    assert control.get_alias_rows() == (AliasRow("qubit", expected),)

    # Alias edits change the global source, not the fixed capture or drafts.
    assert window.plan_panel is plan and window.tunables_panel is knobs
    assert env.parameters is not None and env.parameters.resolve_path("$qubit") == "Q1"
    assert (plan.state.input.data, knobs.state.input.data) == before
    assert (plan.state.base.data, knobs.state.base.data) == bases
    assert edit.text() == "1+" and knobs.editor.local_parse_errors() is not None


def test_prepare_after_applied_alias_captures_new_qubit_defaults(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = window_session.window
    prepared_with_local_text(window_session)
    plan, knobs = window.plan_panel, window.tunables_panel
    edit_qubit_alias(window, qapp, "Q2", apply=True)

    prepare = window.findChild(QPushButton, "workflow.source.use")
    assert prepare is not None
    prepare.click()
    new_env = window.plan_panel.env
    assert new_env.parameters is not None
    assert new_env.parameters.resolve_path("$qubit") == "Q2"
    assert window.plan_panel is not plan and window.tunables_panel is not knobs
    assert window.tunables_panel.env is new_env
    for panel, form in (
        (window.plan_panel, WidePlanForm),
        (window.tunables_panel, WideKnobsForm),
    ):
        assert panel.state.input.data == form.default_input(new_env).data
        assert panel.state.base.data == panel.state.input.data
    assert window.tunables_panel.editor.local_parse_errors() is None
