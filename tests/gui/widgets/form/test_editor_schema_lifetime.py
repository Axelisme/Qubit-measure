"""Captured schema lifetime and one display evaluation per editor commit."""

from collections.abc import Iterator, Mapping
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue
from qtpy.QtCore import QObject
from qtpy.QtGui import QAction
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit
from zcu_tools.gui.form import (
    FormEnv,
    FormModel,
    FormPath,
    FormPreview,
    FormState,
    FormWritePath,
    Option,
    OptionCommand,
    WriteRejected,
)
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._form_context import captured_ext_env


class First(FormModel):
    kind: Literal["first"] = "first"
    value: float = 1.0


class Second(FormModel):
    kind: Literal["second"] = "second"
    value: float = 2.0


class CapturedForm(FormModel):
    amount: float = 1.0
    choice: str = "alpha"
    branch: Annotated[First | Second, Field(discriminator="kind")] = Field(
        default_factory=First
    )

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        allow_beta = (
            env.parameters is not None
            and env.parameters.lookup("general.ext.allow_beta").value
        )
        return {("choice",): ("alpha", "beta") if allow_beta else ("alpha",)}

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        return {
            ("branch",): (
                Option(
                    "First",
                    {"kind": "first"},
                    fields=First,
                    commands=(
                        OptionCommand(
                            "advance", "Advance", lambda env: {"kind": "second"}
                        ),
                    ),
                ),
                Option("Second", {"kind": "second"}, fields=Second),
            )
        }


class RecordingState(FormState[CapturedForm]):
    """Count real calls at the schema and preview interfaces without mocking them."""

    def __init__(self, env: FormEnv) -> None:
        super().__init__(CapturedForm.default_input(env))
        self.schema_reads = 0
        self.preview_reads = 0

    def input_schema(self, env: FormEnv) -> dict[str, JsonValue]:
        self.schema_reads += 1
        return super().input_schema(env)

    def preview(self, env: FormEnv) -> FormPreview:
        self.preview_reads += 1
        return super().preview(env)


@pytest.fixture
def view(
    qapp: QApplication,
) -> Iterator[tuple[FormEnv, RecordingState, FormEditor[CapturedForm]]]:
    env = captured_ext_env(allow_beta=True)
    state = RecordingState(env)
    editor = FormEditor(state, env)
    yield env, state, editor
    editor.detach()
    editor.deleteLater()
    qapp.processEvents()


def control[T: QObject](
    editor: FormEditor[CapturedForm], kind: type[T], name: str
) -> T:
    found = editor.findChild(kind, name)
    assert found is not None
    return found


def commit_text(editor: FormEditor[CapturedForm], text: str) -> None:
    field = control(editor, QLineEdit, "form.input.amount")
    field.setText(text)
    field.editingFinished.emit()


def test_schema_is_reused_until_explicit_refresh(
    view: tuple[FormEnv, RecordingState, FormEditor[CapturedForm]],
) -> None:
    env, state, editor = view
    selector = control(editor, QComboBox, "form.input.choice")
    selector.setCurrentIndex(selector.findData("beta"))
    commit_text(editor, "3")
    action = control(editor, QAction, "form.command.branch.advance")
    action.trigger()
    branch = control(editor, QComboBox, "form.input.branch")
    branch.setCurrentIndex(0)
    assert state.input.data["branch"] == {"kind": "first", "value": 1.0}
    assert state.input.data["amount"] == 3.0
    assert editor.errors() is None
    assert [selector.itemText(i) for i in range(selector.count())] == ["alpha", "beta"]
    assert state.schema_reads == 1

    editor.refresh(captured_ext_env(allow_beta=False))
    selector = control(editor, QComboBox, "form.input.choice")
    assert [selector.itemText(i) for i in range(selector.count())] == [
        "alpha",
        "beta (missing)",
    ]
    errors = editor.errors()
    assert errors is not None and ("choice",) in errors.messages
    assert state.input.data["choice"] == "beta"
    assert state.schema_reads == 2
    editor.refresh(env)
    assert editor.errors() is None
    assert state.schema_reads == 3


@pytest.mark.parametrize("operation", ["text", "branch", "command"])
def test_accepted_commit_evaluates_display_once(
    view: tuple[FormEnv, RecordingState, FormEditor[CapturedForm]], operation: str
) -> None:
    _, state, editor = view
    before = state.preview_reads
    if operation == "text":
        commit_text(editor, "3")
        assert state.input.data["amount"] == 3.0
    elif operation == "branch":
        control(editor, QComboBox, "form.input.branch").setCurrentIndex(1)
        assert state.input.data["branch"] == {"kind": "second", "value": 2.0}
    else:
        control(editor, QAction, "form.command.branch.advance").trigger()
        assert state.input.data["branch"] == {"kind": "second", "value": 2.0}
    assert editor.errors() is None
    assert state.preview_reads - before == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_same_editing_mode_does_not_evaluate_again(
    view: tuple[FormEnv, RecordingState, FormEditor[CapturedForm]], enabled: bool
) -> None:
    _, state, editor = view
    editor.set_editing_enabled(enabled=enabled)
    before = state.preview_reads
    editor.set_editing_enabled(enabled=enabled)
    assert control(editor, QLineEdit, "form.input.amount").isReadOnly() is (not enabled)
    assert state.preview_reads == before


def test_local_error_recovery_and_external_publication_keep_single_evaluation(
    qapp: QApplication,
) -> None:
    env = captured_ext_env(allow_beta=True)
    state = RecordingState(env)
    reject = False

    def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
        if reject:
            raise WriteRejected("Host refused")
        state.write(batch, env)

    editor = FormEditor(
        state,
        env,
        write=write,
        run_command=lambda path, name: state.run_command(path, name, env),
    )
    try:
        commit_text(editor, "-")
        assert editor.local_parse_errors() is not None
        before = state.preview_reads
        commit_text(editor, "1")  # Unchanged state still clears the local error.
        assert editor.local_parse_errors() is None
        assert state.preview_reads - before == 1
        reject = True
        commit_text(editor, "9")
        errors = editor.local_parse_errors()
        assert errors is not None and errors.messages[("amount",)] == "Host refused"
        assert state.input.data["amount"] == 1.0
        state.write({("amount",): 4.0}, env)
        assert editor.local_parse_errors() is not None
        reject = False
        before = state.preview_reads
        commit_text(editor, "4")
        assert editor.errors() is None
        assert state.preview_reads - before == 1
        before = state.preview_reads
        state.write({("amount",): 5.0}, env)
        assert float(control(editor, QLineEdit, "form.input.amount").text()) == 5.0
        assert state.preview_reads - before == 1
        before = state.preview_reads
        commit_text(editor, "6")
        assert state.input.data["amount"] == 6.0
        assert state.preview_reads - before == 1
        editor.detach()
        before = state.preview_reads
        state.write({("amount",): 7.0}, env)
        assert state.preview_reads == before
    finally:
        editor.detach()
        editor.deleteLater()
        qapp.processEvents()
