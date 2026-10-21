"""First expansion builds rows; shape changes leave other rows alone."""

from collections.abc import Iterator, Mapping
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QLineEdit
from zcu_tools.gui.form import (
    Display,
    FormDescription,
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    FormWritePath,
    Option,
    WriteRejected,
)
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._form_converters import RebuildingForm
from tests.gui._user_input import mouse_click, type_text


class Inner(FormModel):
    value: float = Field(default=1.0, gt=0)


class First(FormModel):
    kind: Literal["first"] = "first"
    one: float = 1.0


class Second(FormModel):
    kind: Literal["second"] = "second"
    two: float = 2.0


class LazyForm(RebuildingForm):
    branch_models = {("branch",): (First, Second)}

    stable: float = 5.0
    folded: Annotated[Inner, Display(collapsed=True)] = Field(default_factory=Inner)
    advanced: Annotated[float, Display(group="Advanced")] = 3.0
    branch: Annotated[
        First | Second, Field(discriminator="kind"), Display(collapsed=False)
    ] = Field(default_factory=First)
    optional: Annotated[Inner | None, Display(collapsed=False)] = Field(
        default_factory=Inner
    )


@pytest.fixture
def view(
    qapp: QApplication,
) -> Iterator[tuple[FormEditor[LazyForm], FormState[LazyForm]]]:
    env = FormEnv()
    state = FormState(
        FormInput(
            LazyForm,
            {
                "stable": 5.0,
                "folded": {"value": 1.0},
                "advanced": 3.0,
                "branch": {"kind": "first", "one": 1.0},
                "optional": {"value": 1.0},
            },
        )
    )
    editor = FormEditor(state, env)
    yield editor, state
    editor.detach()
    editor.deleteLater()
    qapp.processEvents()


def test_expanding_an_empty_selected_layer_removes_the_fold_title(
    qapp: QApplication,
) -> None:
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


@pytest.mark.parametrize(
    "path,fold_path", [("folded.value", "folded"), ("advanced", "group.Advanced")]
)
def test_first_expansion_builds_controls_and_collapse_keeps_uncommitted_text(
    view: tuple[FormEditor[LazyForm], FormState[LazyForm]],
    path: str,
    fold_path: str,
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


@pytest.mark.parametrize(
    "path,fold_path", [("folded.value", "folded"), ("advanced", "group.Advanced")]
)
@pytest.mark.parametrize("draft", ["12.5", "unfinished -"])
def test_collapsing_focused_text_neither_commits_nor_parses_it(
    view: tuple[FormEditor[LazyForm], FormState[LazyForm]],
    qapp: QApplication,
    path: str,
    fold_path: str,
    draft: str,
) -> None:
    editor, state = view
    editor.resize(700, 900)
    editor.show()
    qapp.processEvents()
    fold = title(editor, fold_path)
    mouse_click(fold, fold.rect().center())
    qapp.processEvents()
    field = text(editor, path)
    field.setFocus(Qt.FocusReason.TabFocusReason)
    assert field.hasFocus()
    field.selectAll()
    type_text(field, draft)
    before = state.input.data
    mouse_click(fold, fold.rect().center())
    qapp.processEvents()
    assert not field.isVisibleTo(editor)
    assert state.input.data == before
    assert editor.local_parse_errors() is None
    mouse_click(fold, fold.rect().center())
    qapp.processEvents()
    assert text(editor, path) is field
    assert field.isVisibleTo(editor)
    assert field.text() == draft


def select_unless_busy(
    state: FormState[LazyForm],
    env: FormEnv,
    busy: bool,
    path: FormWritePath,
    selection: JsonValue,
) -> None:
    """Share the host busy policy with the explicit selection callback."""
    if busy:
        raise WriteRejected("Host busy")
    state.select(path, selection, env)


@pytest.mark.parametrize("reject", [False, True])
def test_restored_focused_draft_commits_once_on_normal_focus_loss(
    qapp: QApplication,
    reject: bool,
) -> None:
    env = FormEnv()
    state = FormState(LazyForm.default_input(env))
    batches: list[dict[FormWritePath, JsonValue]] = []
    publications: list[int] = []

    def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
        batches.append(dict(batch))
        if reject:
            raise WriteRejected("Host busy")
        state.write(batch, env)

    unsubscribe = state.subscribe(lambda: publications.append(1))
    editor = FormEditor(
        state,
        env,
        write=write,
        select=lambda path, selection: select_unless_busy(
            state, env, reject, path, selection
        ),
        run_command=lambda _path, _name: None,
    )
    try:
        editor.show()
        qapp.processEvents()
        fold = title(editor, "folded")
        mouse_click(fold, fold.rect().center())
        field = text(editor, "folded.value")
        field.setFocus()
        assert field.hasFocus()
        field.selectAll()
        type_text(field, "12.5")
        before = state.input.data
        mouse_click(fold, fold.rect().center())
        qapp.processEvents()
        assert batches == [] and publications == []
        assert state.input.data == before
        mouse_click(fold, fold.rect().center())
        field.setFocus()
        assert field.hasFocus()
        text(editor, "stable").setFocus()
        qapp.processEvents()
        assert batches == [{("folded", "value"): 12.5}]
        assert field.text() == "12.5"
        if reject:
            assert state.input.data == before
            assert publications == []
            errors = editor.local_parse_errors()
            assert errors is not None and errors.messages == {
                ("folded", "value"): "Host busy"
            }
        else:
            assert state.input.data["folded"] == {"value": 12.5}
            assert publications == [1]
    finally:
        unsubscribe()
        editor.detach()
        editor.deleteLater()
        qapp.processEvents()


def test_replacing_a_focused_branch_does_not_commit_retired_text(
    qapp: QApplication,
) -> None:
    env = FormEnv()
    state = FormState(LazyForm.default_input(env))
    batches: list[dict[FormWritePath, JsonValue]] = []
    editor = FormEditor(
        state,
        env,
        write=lambda batch: batches.append(dict(batch)),
        select=lambda path, selection: state.select(path, selection, env),
        run_command=lambda _path, _name: None,
    )
    try:
        editor.show()
        qapp.processEvents()
        field = text(editor, "branch.one")
        field.setFocus()
        assert field.hasFocus()
        field.selectAll()
        type_text(field, "7")
        state.select(("branch",), {"kind": "second"}, env)
        qapp.processEvents()
        assert batches == []
        assert state.input.data["branch"] == {"kind": "second", "two": 2.0}
        assert float(text(editor, "branch.two").text()) == 2.0
    finally:
        editor.detach()
        editor.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("operation", ["option", "nullable"])
def test_shape_change_keeps_unrelated_widgets_and_fold_state(
    view: tuple[FormEditor[LazyForm], FormState[LazyForm]],
    operation: str,
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
