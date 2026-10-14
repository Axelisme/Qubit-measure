"""String Expression actions write envelopes and display core-resolved names."""

from collections.abc import Iterator

import pytest
from PyQt6.QtGui import QAction
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class StringForm(FormModel):
    """A free text field resolves string expressions."""

    name: str = "literal"


@pytest.fixture
def name_editor(
    qapp: QApplication,
) -> Iterator[tuple[FormEditor[StringForm], FormState[StringForm], FormEnv]]:
    """Own a view over a memory-only fixed alias capture on the existing app."""
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            general=PointGeneral(created_at="2026-10-11T00:00:00Z"),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=SavedPoint(entry_id="entry", point="synthetic"),
        roles=None,
        storage=None,
    ).bind_aliases((AliasRow("device", "yoko1"), AliasRow("unbound", "")), {})
    env = FormEnv(parameters=parameters)
    state = FormState(StringForm.default_input(env))
    editor = FormEditor(state, env)
    yield editor, state, env
    editor.close()
    editor.deleteLater()
    qapp.processEvents()


def test_string_mode_writes_expression_and_projects_preview_and_errors(
    name_editor: tuple[FormEditor[StringForm], FormState[StringForm], FormEnv],
) -> None:
    editor, state, env = name_editor
    field = "name"
    mode = editor.findChild(QAction, f"form.mode.{field}")
    assert isinstance(mode, QAction)
    mode.trigger()
    text = editor.findChild(QLineEdit, f"form.input.{field}")
    assert isinstance(text, QLineEdit)
    text.setText("$device")
    assert state.input.data[field] == {"expr": "$device"}
    assert state.preview(env).values[(field,)] == "yoko1"
    shown = editor.findChild(QLabel, f"form.value.{field}")
    assert isinstance(shown, QLabel)
    assert '$device = "yoko1"' in shown.text()
    text.setText("$unbound")
    error = editor.findChild(QLabel, f"form.error.{field}")
    assert isinstance(error, QLabel)
    assert "unbound" in error.text()
    assert shown.text() == ""
    assert editor.errors() is not None
    # Hydration retains expression mode and never replaces authored input.
    state.reset(FormInput(StringForm, {field: {"expr": "$device"}}))
    assert mode.isChecked()
    assert text.text() == "$device"
    assert state.input.data[field] == {"expr": "$device"}
    mode.trigger()
    text.setText("$device")
    assert state.input.data[field] == "$device"
    assert state.preview(env).values[(field,)] == "$device"


def test_string_direct_mode_fills_the_resolved_name(
    name_editor: tuple[FormEditor[StringForm], FormState[StringForm], FormEnv],
) -> None:
    editor, state, env = name_editor
    state.write({("name",): {"expr": "$device"}}, env)
    mode = editor.findChild(QAction, "form.mode.name")
    assert isinstance(mode, QAction)

    mode.trigger()

    assert not mode.isChecked()
    assert state.input.data["name"] == "yoko1"
    text = editor.findChild(QLineEdit, "form.input.name")
    assert isinstance(text, QLineEdit)
    assert text.text() == "yoko1"


def test_string_direct_mode_keeps_an_unresolved_expression_as_literal_text(
    name_editor: tuple[FormEditor[StringForm], FormState[StringForm], FormEnv],
) -> None:
    editor, state, env = name_editor
    state.write({("name",): {"expr": "$unbound"}}, env)
    mode = editor.findChild(QAction, "form.mode.name")
    assert isinstance(mode, QAction)

    mode.trigger()

    assert not mode.isChecked()
    assert state.input.data["name"] == "$unbound"
    assert editor.errors() is None
