"""Text rows commit typed text on Enter or when the field loses focus."""

from collections.abc import Callable, Iterator, Mapping
from typing import Annotated

import pytest
from pydantic import Field, JsonValue
from PyQt6.QtGui import QAction
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormModel,
    FormState,
    FormWritePath,
    WriteRejected,
)
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import key_click
from tests.gui._user_input import type_text as type_keys


class CommitForm(FormModel):
    """Every text row kind, a decimals row and an unfilled nullable row."""

    amount: float = Field(default=2.0, gt=0)
    count: int = 2
    name: str = "a"
    amplitude: complex = 1 + 0j
    delay: Annotated[float, Display(decimals=2)] = 1.23456
    spare: float | None = None


type Batch = dict[FormWritePath, JsonValue]
type View = tuple[FormEditor[CommitForm], FormState[CommitForm], list[Batch]]
type Build = Callable[[Callable[[Batch], None] | None], View]


@pytest.fixture
def build(qapp: QApplication) -> Iterator[Build]:
    """Build shown editors whose host write records every batch it receives.

    The optional check runs on each batch before it is recorded and installed;
    it may raise WriteRejected to refuse the batch.
    """
    editors: list[FormEditor[CommitForm]] = []

    def make(check: Callable[[Batch], None] | None) -> View:
        env = FormEnv()
        state = FormState(CommitForm.default_input(env))
        batches: list[Batch] = []

        def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
            if check is not None:
                check(dict(batch))
            batches.append(dict(batch))
            state.write(batch, env)

        def run_command(path: FormWritePath, name: str) -> None:
            state.run_command(path, name, env)

        editor = FormEditor(state, env, write=write, run_command=run_command)
        editors.append(editor)
        editor.show()
        editor.activateWindow()
        qapp.processEvents()
        return editor, state, batches

    yield make
    for editor in editors:
        editor.close()
        editor.deleteLater()
    qapp.processEvents()


def line(view: QWidget, path: str) -> QLineEdit:
    """Find the documented text input of a dotted model-field path."""
    widget = view.findChild(QLineEdit, f"form.input.{path}")
    assert isinstance(widget, QLineEdit), path
    return widget


def label(view: QWidget, name: str) -> QLabel:
    """Find a documented value or error label by its complete object name."""
    widget = view.findChild(QLabel, name)
    assert isinstance(widget, QLabel), name
    return widget


def type_text(edit: QLineEdit, text: str) -> None:
    """Replace edit's whole text by typing text key by key, without Enter."""
    edit.selectAll()
    type_keys(edit, text)


def press_enter(edit: QLineEdit) -> None:
    """Press Enter in edit."""
    key_click(edit, Qt.Key.Key_Return)


def focus(widget: QWidget) -> None:
    """Give widget keyboard focus as a mouse click would."""
    widget.setFocus(Qt.FocusReason.MouseFocusReason)
    QApplication.processEvents()
    assert widget.hasFocus()


@pytest.mark.parametrize(
    ("path", "typed", "committed"),
    [
        ("amount", "3.5", 3.5),
        ("count", "7", 7),
        ("name", "bc", "bc"),
        ("amplitude", "2+3j", {"__complex__": [2.0, 3.0]}),
    ],
)
def test_typed_text_reaches_state_once_enter_commits_it(
    build: Build, path: str, typed: str, committed: JsonValue
) -> None:
    view, state, batches = build(None)
    validity: list[bool] = []
    view.validity_changed.connect(validity.append)
    before = state.input.data
    edit = line(view, path)

    type_text(edit, typed)

    assert batches == []
    assert state.input.data == before
    assert edit.text() == typed
    assert validity == []

    press_enter(edit)

    assert batches == [{(path,): committed}]
    assert state.input.data[path] == committed
    # Enter again on the same text writes nothing more.
    press_enter(edit)
    assert batches == [{(path,): committed}]


def test_leaving_a_field_commits_its_text_once(build: Build) -> None:
    view, state, batches = build(None)
    amount = line(view, "amount")
    count = line(view, "count")
    focus(amount)

    type_text(amount, "4.5")
    assert batches == []

    focus(count)

    assert batches == [{("amount",): 4.5}]
    assert state.input.data["amount"] == 4.5
    # Entering and leaving again without typing writes nothing.
    focus(amount)
    focus(count)
    assert batches == [{("amount",): 4.5}]


def test_enter_or_leaving_without_typing_writes_nothing(build: Build) -> None:
    view, state, batches = build(None)
    before = state.input.data
    assert line(view, "delay").text() == "1.23"
    assert line(view, "spare").text() == ""
    for path in ("amount", "delay", "spare"):
        edit = line(view, path)
        focus(edit)
        press_enter(edit)
        focus(line(view, "count"))
    assert batches == []
    # The decimals row's rounded text never replaces the state's digits, and
    # the unfilled row stays unfilled.
    assert state.input.data == before
    assert "spare" not in state.input.data


def test_unparseable_text_shows_its_error_only_once_committed(build: Build) -> None:
    view, state, batches = build(None)
    validity: list[bool] = []
    view.validity_changed.connect(validity.append)
    count = line(view, "count")
    error = label(view, "form.error.count")

    type_text(count, "x")

    assert view.local_parse_errors() is None
    assert error.isHidden()
    assert validity == []

    press_enter(count)

    local = view.local_parse_errors()
    assert local is not None
    assert set(local.messages) == {("count",)}
    assert not error.isHidden()
    assert validity == [False]
    assert batches == []
    assert state.input.data["count"] == 2

    type_text(count, "2")
    press_enter(count)

    assert view.local_parse_errors() is None
    assert error.isHidden()
    assert validity == [False, True]
    assert state.input.data["count"] == 2


def test_expression_text_commits_its_envelope_on_enter(build: Build) -> None:
    view, state, batches = build(None)
    mode = view.findChild(QAction, "form.mode.amount")
    assert isinstance(mode, QAction)
    amount = line(view, "amount")

    # Switching the mode writes the current text in the new mode at once.
    mode.trigger()
    assert batches == [{("amount",): {"expr": "2.0"}}]

    type_text(amount, "1 / 4")

    assert state.input.data["amount"] == {"expr": "2.0"}
    assert len(batches) == 1

    press_enter(amount)

    assert batches[-1] == {("amount",): {"expr": "1 / 4"}}
    assert state.input.data["amount"] == {"expr": "1 / 4"}
    assert label(view, "form.value.amount").text() == "0.25"


def test_rejected_text_stays_local_until_an_accepted_commit(build: Build) -> None:
    def reject_large(batch: Batch) -> None:
        value = batch.get(("amount",))
        if isinstance(value, float) and value > 10:
            raise WriteRejected("Amount is too large")

    view, state, batches = build(reject_large)
    amount = line(view, "amount")

    type_text(amount, "12")
    assert batches == []

    press_enter(amount)

    assert batches == []
    assert state.input.data["amount"] == 2.0
    assert amount.text() == "12"
    assert label(view, "form.error.amount").text() == "Amount is too large"
    local = view.local_parse_errors()
    assert local is not None
    assert set(local.messages) == {("amount",)}

    type_text(amount, "5")
    press_enter(amount)

    assert batches == [{("amount",): 5.0}]
    assert view.local_parse_errors() is None
    assert state.input.data["amount"] == 5.0


def test_read_only_drops_pending_text_and_commits_nothing(build: Build) -> None:
    view, state, batches = build(None)
    amount = line(view, "amount")

    type_text(amount, "9")
    view.set_editing_enabled(enabled=False)

    assert amount.isReadOnly()
    assert amount.text() == "2.0"
    press_enter(amount)
    type_text(amount, "7")
    press_enter(amount)
    assert batches == []

    view.set_editing_enabled(enabled=True)

    assert amount.text() == "2.0"
    assert batches == []
    assert state.input.data["amount"] == 2.0


def test_detach_with_pending_text_writes_nothing(build: Build) -> None:
    view, state, batches = build(None)
    amount = line(view, "amount")
    focus(amount)
    type_text(amount, "9")

    view.detach()

    assert batches == []
    assert state.input.data["amount"] == 2.0
