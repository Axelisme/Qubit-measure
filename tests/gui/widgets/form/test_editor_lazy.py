"""First expansion builds rows; shape changes leave other rows alone."""

from collections.abc import Iterator, Mapping
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue
from qtpy.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QLineEdit
from zcu_tools.gui.form import (
    Display, FormDescription, FormEnv, FormInput, FormModel, FormPath, FormState, FormWritePath, Option,
)
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import mouse_click


class Inner(FormModel):
    value: float = Field(default=1.0, gt=0)


class First(FormModel):
    kind: Literal["first"] = "first"
    one: float = 1.0


class Second(FormModel):
    kind: Literal["second"] = "second"
    two: float = 2.0


class LazyForm(FormModel):
    stable: float = 5.0
    folded: Annotated[Inner, Display(collapsed=True)] = Field(default_factory=Inner)
    advanced: Annotated[float, Display(group="Advanced")] = 3.0
    branch: Annotated[First | Second, Field(discriminator="kind")] = Field(
        default_factory=First
    )
    optional: Inner | None = Field(default_factory=Inner)


@pytest.fixture
def view(qapp: QApplication) -> Iterator[tuple[FormEditor[LazyForm], FormState[LazyForm]]]:
    env = FormEnv()
    state = FormState(FormInput(LazyForm, {
        "stable": 5.0, "folded": {"value": -1.0}, "advanced": 3.0,
        "branch": {"kind": "first", "one": 1.0}, "optional": {"value": 1.0},
    }))
    editor = FormEditor(state, env)
    yield editor, state
    editor.detach()
    editor.deleteLater()
    qapp.processEvents()


def test_expanding_an_empty_selected_layer_removes_the_fold_title(qapp: QApplication) -> None:
    class Empty(FormModel):
        kind: Literal["empty"] = "empty"

    class FixedBranch(FormModel):
        branch: Annotated[Empty, Display(collapsed=True)] = Field(default_factory=Empty)

        @classmethod
        def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
            return {("branch",): (Option("Fixed", {"kind": "empty"}, fields=Empty),)}

    env = FormEnv()
    state = FormState(FixedBranch.default_input(env))
    editor = FormEditor(state, env)
    try:
        before: dict[str, JsonValue] = state.input.data
        fold = editor.findChild(QLabel, "form.fold.branch")
        assert fold is not None
        mouse_click(fold, fold.rect().center())
        assert editor.findChild(QLabel, "form.fold.branch") is None
        assert state.input.data == before
        assert editor.errors() is None
    finally:
        editor.detach()
        editor.deleteLater()
        qapp.processEvents()


class DescribedState(FormState[LazyForm]):
    """Observe requested public layers without replacing the core's behaviour."""

    def __init__(self, env: FormEnv) -> None:
        super().__init__(LazyForm.default_input(env))
        self.requested: set[FormWritePath] = set()

    def describe(self, path: FormWritePath, env: FormEnv) -> FormDescription:
        self.requested.add(path)
        return super().describe(path, env)


def test_model_description_is_requested_on_first_expansion(qapp: QApplication) -> None:
    env = FormEnv()
    state = DescribedState(env)
    editor = FormEditor(state, env)
    try:
        assert ("folded",) not in state.requested
        fold = title(editor, "folded")
        mouse_click(fold, fold.rect().center())
        assert ("folded",) in state.requested
        assert float(text(editor, "folded.value").text()) == 1.0
    finally:
        editor.detach()
        editor.deleteLater()
        qapp.processEvents()


def title(editor: FormEditor[LazyForm], path: str) -> QLabel:
    found = editor.findChild(QLabel, "form.fold." + path)
    assert found is not None
    return found


def text(editor: FormEditor[LazyForm], path: str) -> QLineEdit:
    found = editor.findChild(QLineEdit, "form.input." + path)
    assert found is not None
    return found


@pytest.mark.parametrize("path,fold_path", [("folded.value", "folded"), ("advanced", "group.Advanced")])
def test_first_expansion_builds_controls_and_collapse_keeps_uncommitted_text(
    view: tuple[FormEditor[LazyForm], FormState[LazyForm]], path: str, fold_path: str,
) -> None:
    editor, state = view
    before = state.input.data
    assert editor.findChild(QLineEdit, "form.input." + path) is None
    fold = title(editor, fold_path)
    mouse_click(fold, fold.rect().center())
    field = text(editor, path)
    field.setText("unfinished -")
    mouse_click(fold, fold.rect().center())
    mouse_click(fold, fold.rect().center())
    assert text(editor, path) is field
    assert field.text() == "unfinished -"
    assert state.input.data == before


@pytest.mark.parametrize("operation", ["option", "nullable"])
def test_shape_change_keeps_unrelated_widgets_and_fold_state(
    view: tuple[FormEditor[LazyForm], FormState[LazyForm]], operation: str,
) -> None:
    editor, _ = view
    stable = text(editor, "stable")
    fold = title(editor, "folded")
    mouse_click(fold, fold.rect().center())
    inner = text(editor, "folded.value")
    if operation == "option":
        selector = editor.findChild(QComboBox, "form.input.branch")
        assert selector is not None
        selector.setCurrentIndex(1)
        assert float(text(editor, "branch.two").text()) == 2.0
    else:
        enabled = editor.findChild(QCheckBox, "form.enabled.optional")
        assert enabled is not None
        enabled.setChecked(False)
    assert text(editor, "stable") is stable
    assert title(editor, "folded") is fold
    assert text(editor, "folded.value") is inner
    assert fold.text().startswith("▾ ")
