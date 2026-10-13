"""A host write callable takes the editor's parsed input and may reject it."""

from collections.abc import Callable, Iterator, Mapping

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue
from qtpy.QtWidgets import QApplication, QCheckBox, QLabel, QLineEdit, QWidget
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormModel,
    FormState,
    FormWritePath,
    WriteRejected,
)
from zcu_tools.gui.widgets.form import FormEditor


class Tone(BaseModel):
    """The inline model of a key-or-inline field."""

    model_config = ConfigDict(extra="forbid")

    gain: float = 0.5


class Section(FormModel):
    """An optional nested section toggled by its Enabled box."""

    start: float = 1.0


class HostForm(FormModel):
    """Text, key, numeric, bool and optional-section inputs a host can see."""

    note: str = ""
    tone: str | Tone = ""
    amount: float = 1.0
    flag: bool = False
    section: Section | None = None


type Editor = FormEditor[HostForm]
type Write = Callable[[Mapping[FormWritePath, JsonValue]], None]
type Build = Callable[[FormState[HostForm], Write | None], Editor]


@pytest.fixture
def build(qapp: QApplication) -> Iterator[Build]:
    """Build editors over a state, optionally with a host write; delete them later."""
    editors: list[Editor] = []

    def make(state: FormState[HostForm], write: Write | None) -> Editor:
        editor = (
            FormEditor(state, FormEnv())
            if write is None
            else FormEditor(state, FormEnv(), write=write)
        )
        editors.append(editor)
        return editor

    yield make
    for editor in editors:
        editor.close()
        editor.deleteLater()
    qapp.processEvents()


def child[W: QWidget](editor: Editor, kind: type[W], name: str) -> W:
    """Find a documented public child by its complete object name."""
    widget = editor.findChild(kind, name)
    assert isinstance(widget, kind), name
    return widget


def new_state() -> FormState[HostForm]:
    """A draft holding the form's defaults."""
    return FormState(HostForm.default_input(FormEnv()))


def test_without_write_parsed_input_goes_to_the_state(build: Build) -> None:
    state = new_state()
    editor = build(state, None)
    child(editor, QLineEdit, "form.input.amount").setText("3.5")
    child(editor, QCheckBox, "form.input.flag").setChecked(True)
    assert state.input.data["amount"] == 3.5
    assert state.input.data["flag"] is True
    assert editor.errors() is None


def test_write_receives_every_parsed_batch_and_the_editor_still_shows_state(
    build: Build,
) -> None:
    state = new_state()
    before = state.input.data
    received: list[dict[FormWritePath, JsonValue]] = []
    editor = build(state, lambda batch: received.append(dict(batch)))
    child(editor, QLineEdit, "form.input.amount").setText("3.5")
    child(editor, QCheckBox, "form.input.flag").setChecked(True)
    assert received == [{("amount",): 3.5}, {("flag",): True}]
    # This host installs nothing, and the editor does not write the state itself.
    assert state.input.data == before

    state.write({("amount",): 7.0}, FormEnv())
    assert float(child(editor, QLineEdit, "form.input.amount").text()) == 7.0
    assert editor.errors() is None


@pytest.mark.parametrize("path", ["note", "tone"])
def test_rejected_text_stays_typed_with_the_host_message_until_accepted(
    build: Build, path: str
) -> None:
    state = new_state()

    def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
        for value in batch.values():
            if isinstance(value, str) and value.startswith("@"):
                raise WriteRejected(f"Unknown capture key {value}")
        state.write(batch, FormEnv())

    editor = build(state, write)
    observed: list[bool] = []
    editor.validity_changed.connect(observed.append)
    text = child(editor, QLineEdit, f"form.input.{path}")
    error = child(editor, QLabel, f"form.error.{path}")
    text.setText("@{gain}")
    assert text.text() == "@{gain}"
    assert state.input.data[path] == ""
    assert error.text() == "Unknown capture key @{gain}"
    local = editor.local_parse_errors()
    assert isinstance(local, FormErrors)
    assert local.messages == {(path,): "Unknown capture key @{gain}"}
    assert observed == [False]

    # An accepted write elsewhere and a refresh redraw state; the text stays.
    child(editor, QLineEdit, "form.input.amount").setText("2")
    editor.refresh(FormEnv())
    assert state.input.data["amount"] == 2.0
    assert text.text() == "@{gain}"
    assert error.text() == "Unknown capture key @{gain}"

    text.setText("plain")
    assert state.input.data[path] == "plain"
    assert error.text() == ""
    assert editor.errors() is None
    assert observed == [False, True]


def test_rejected_checkbox_and_section_show_state_with_the_host_message(
    build: Build,
) -> None:
    state = new_state()
    before = state.input.data

    def write(_batch: Mapping[FormWritePath, JsonValue]) -> None:
        raise WriteRejected("Locked by the host")

    editor = build(state, write)
    flag = child(editor, QCheckBox, "form.input.flag")
    flag.setChecked(True)
    enabled = child(editor, QCheckBox, "form.enabled.section")
    enabled.setChecked(True)
    editor.refresh(FormEnv())
    assert state.input.data == before
    assert not flag.isChecked()
    assert not enabled.isChecked()
    assert child(editor, QLabel, "form.error.flag").text() == "Locked by the host"
    assert child(editor, QLabel, "form.error.section").text() == "Locked by the host"
    local = editor.local_parse_errors()
    assert isinstance(local, FormErrors)
    assert set(local.messages) == {("flag",), ("section",)}

    editor.discard_local_text()
    assert editor.errors() is None
    assert child(editor, QLabel, "form.error.flag").text() == ""
