"""String mode controls write envelopes and display core-resolved names."""

from collections.abc import Iterator, Mapping

import pytest
from pydantic import JsonValue
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormPath, FormState
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class StringForm(FormModel):
    """Both free text and finite choices can resolve string expressions."""

    name: str = "literal"
    choice: str = "yoko1"

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Choices constrain ordinary resolved values rather than input text."""
        return {("choice",): ("yoko1",)}


@pytest.fixture
def name_editor(
    qapp: QApplication,
) -> Iterator[tuple[FormEditor[StringForm], FormState[StringForm], FormEnv]]:
    """Own a view over a memory-only fixed alias capture on the existing app."""
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
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


@pytest.mark.parametrize("field", ["name", "choice"])
def test_string_mode_writes_expression_and_projects_preview_and_errors(
    name_editor: tuple[FormEditor[StringForm], FormState[StringForm], FormEnv],
    field: str,
) -> None:
    editor, state, env = name_editor
    mode = editor.findChild(QComboBox, f"form.mode.{field}")
    assert isinstance(mode, QComboBox)
    mode.setCurrentIndex(mode.findData("expr"))
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
    assert mode.currentData() == "expr"
    assert text.text() == "$device"
    assert state.input.data[field] == {"expr": "$device"}
    if field == "name":
        mode.setCurrentIndex(mode.findData("direct"))
        text.setText("$device")
        assert state.input.data[field] == "$device"
        assert state.preview(env).values[(field,)] == "$device"
