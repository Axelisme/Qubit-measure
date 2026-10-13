"""Display groups fold a model level's grouped rows into one block per group."""

from collections.abc import Callable, Iterator
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)
from zcu_tools.gui.form import Display, FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor

type View = tuple[FormEditor[FormModel], FormState[FormModel]]
type Build = Callable[[type[FormModel], dict[str, JsonValue] | None], View]

ERROR_NOTE = "Contains errors"


@pytest.fixture
def build(qapp: QApplication) -> Iterator[Build]:
    """Build editors over given input (or default input) and delete them later."""
    editors: list[FormEditor[FormModel]] = []

    def make(model: type[FormModel], data: dict[str, JsonValue] | None) -> View:
        env = FormEnv()
        initial = model.default_input(env) if data is None else FormInput(model, data)
        state = FormState(initial)
        editor = FormEditor(state, env)
        editors.append(editor)
        return editor, state

    yield make
    for editor in editors:
        editor.close()
        editor.deleteLater()
    qapp.processEvents()


def child[W: QWidget](view: QWidget, kind: type[W], name: str) -> W:
    """Find a documented public child by its complete object name."""
    widget = view.findChild(kind, name)
    assert isinstance(widget, kind), name
    return widget


def direct[W: QWidget](parent: QWidget, kind: type[W]) -> list[W]:
    """Children of kind whose parent widget is parent itself."""
    return parent.findChildren(kind, "", Qt.FindChildOption.FindDirectChildrenOnly)


def group_fold(view: QWidget, name: str) -> QPushButton:
    """The Expand/Collapse button of the group block form.group.<name>."""
    (button,) = direct(child(view, QWidget, f"form.group.{name}"), QPushButton)
    return button


def model_fold(view: QWidget, path: str) -> QPushButton:
    """The Expand/Collapse button in the header of the model row at path."""
    (button,) = [
        button
        for button in direct(row_of(view, path), QPushButton)
        if button.text() in ("Expand", "Collapse")
    ]
    return button


def error_note(view: QWidget, name: str) -> QLabel:
    """The block header's errors note of form.group.<name>, shown or not."""
    block = child(view, QWidget, f"form.group.{name}")
    (note,) = [label for label in direct(block, QLabel) if label.text() == ERROR_NOTE]
    return note


def row_of(view: QWidget, path: str) -> QWidget:
    """The row widget at path: the parent of its form.value.<path> label."""
    row = child(view, QLabel, f"form.value.{path}").parentWidget()
    assert row is not None
    return row


def shown(view: QWidget, path: str) -> bool:
    """Whether the input at path would be visible while view is shown."""
    return child(view, QWidget, f"form.input.{path}").isVisibleTo(view)


def top(view: QWidget, widget: QWidget) -> int:
    """Widget's top edge in view coordinates."""
    return widget.mapTo(view, QPoint(0, 0)).y()


class Section(BaseModel):
    """An ordinary nested model with one positive leaf."""

    model_config = ConfigDict(extra="forbid")

    level: float = Field(default=1.0, gt=0)


class Drive(BaseModel):
    """A nested level mixing ungrouped rows, grouped rows and a grouped model."""

    model_config = ConfigDict(extra="forbid")

    ch: int = 0
    phase: Annotated[float, Display(group="Advanced")] = 0.0
    freq: float = 100.0
    inner: Annotated[Section, Display(group="Advanced")] = Field(
        default_factory=Section
    )


class GroupForm(FormModel):
    """Root rows: groups interleave with ungrouped rows in declaration order."""

    first: Annotated[float, Display(group="Advanced")] = 1.0
    mode: Literal["fast", "slow"] = "fast"
    window: Annotated[float, Display(group="Timing")] = 2.0
    second: Annotated[float, Display(group="Advanced")] = 3.0
    after: float = 4.0
    drive: Drive = Field(default_factory=Drive)
    secret: Annotated[float, Display(group="Hidden", hidden=True)] = 5.0
    kept: Annotated[float, Display(group="Partly", hidden=True)] = 6.0
    visible: Annotated[float, Display(group="Partly")] = 7.0


def test_groups_fold_grouped_rows_after_ungrouped_rows(
    build: Build, qapp: QApplication
) -> None:
    view, state = build(GroupForm, None)
    before = dict(state.input.data)
    members = {
        "Advanced": ("first", "second"),
        "Timing": ("window",),
        "Partly": ("visible",),
        "drive.Advanced": ("drive.phase", "drive.inner", "drive.inner.level"),
    }
    for name, paths in members.items():
        block = child(view, QWidget, f"form.group.{name}")
        assert group_fold(view, name).text() == "Expand", name
        for path in paths:
            assert block.isAncestorOf(row_of(view, path)), path
            # Every group starts folded.
            assert not row_of(view, path).isVisibleTo(view), path
    for path in ("mode", "after", "drive.ch", "drive.freq"):
        assert shown(view, path), path
        assert not child(view, QWidget, "form.group.Advanced").isAncestorOf(
            child(view, QWidget, f"form.input.{path}")
        ), path
    # A group whose every member is hidden has no block.
    assert view.findChild(QWidget, "form.group.Hidden") is None
    assert view.findChildren(QWidget, "form.input.kept") == []

    for name in members:
        group_fold(view, name).click()
    view.resize(600, view.sizeHint().height())
    view.show()
    qapp.processEvents()
    for name, paths in members.items():
        assert group_fold(view, name).text() == "Collapse", name
        assert all(row_of(view, path).isVisible() for path in paths), name
    # Ungrouped rows first, then groups by first appearance, members in order.
    order = [
        top(view, row_of(view, path))
        for path in (
            "mode",
            "after",
            "drive.ch",
            "drive.freq",
            "drive.phase",
            "drive.inner",
            "first",
            "second",
            "window",
            "visible",
        )
    ]
    assert order == sorted(order), order
    assert len(set(order)) == len(order), order
    assert top(view, child(view, QWidget, "form.group.Advanced")) > top(
        view, child(view, QWidget, "form.input.drive.freq")
    )
    # A group title sits in its level's label column like a model row label:
    # its fold button starts where the level's controls start.
    fold_left = group_fold(view, "Advanced").mapTo(view, QPoint(0, 0)).x()
    mode_left = child(view, QComboBox, "form.input.mode").mapTo(view, QPoint(0, 0))
    assert fold_left == mode_left.x()
    assert state.input.data == before


class Alpha(BaseModel):
    """First branch; its Advanced group shares a name with Beta's."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["alpha"] = "alpha"
    x: float = 1.0
    phase: Annotated[float, Display(group="Advanced")] = 0.0


class Beta(BaseModel):
    """Second branch with a different ungrouped row."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["beta"] = "beta"
    y: float = 2.0
    phase: Annotated[float, Display(group="Advanced")] = 0.0


class FoldForm(FormModel):
    """Root groups, a tagged branch with a group, and two model folds."""

    gain: float = 1.0
    delay: Annotated[float, Display(group="Advanced")] = 0.0
    window: Annotated[float, Display(group="Timing")] = 0.0
    shape: Annotated[Alpha | Beta, Field(discriminator="kind")] = Field(
        default_factory=Alpha
    )
    drive: Section = Field(default_factory=Section)
    extra: Annotated[Section, Display(collapsed=True)] = Field(default_factory=Section)


def switch_shape(view: QWidget, tag: str) -> None:
    """Select the shape branch tag, which rebuilds the editor's rows."""
    selector = child(view, QComboBox, "form.input.shape.kind")
    selector.setCurrentIndex(selector.findData(tag))


def test_user_folds_survive_a_rebuild_of_the_same_editor(build: Build) -> None:
    view, state = build(FoldForm, None)
    assert not shown(view, "delay")
    assert not shown(view, "shape.phase")
    assert shown(view, "drive.level")
    assert not shown(view, "extra.level")

    group_fold(view, "Advanced").click()
    group_fold(view, "shape.Advanced").click()
    model_fold(view, "drive").click()
    model_fold(view, "extra").click()
    switch_shape(view, "beta")

    shape = state.input.data["shape"]
    assert isinstance(shape, dict)
    assert shape["kind"] == "beta"
    assert view.findChild(QLineEdit, "form.input.shape.y") is not None
    assert shown(view, "delay")
    assert shown(view, "shape.phase")
    assert group_fold(view, "Advanced").text() == "Collapse"
    assert not shown(view, "drive.level")
    assert model_fold(view, "drive").text() == "Expand"
    assert shown(view, "extra.level")
    # Untouched groups keep their folded default.
    assert not shown(view, "window")

    # Folding back is remembered as well.
    group_fold(view, "Advanced").click()
    switch_shape(view, "alpha")
    assert view.findChild(QLineEdit, "form.input.shape.x") is not None
    assert not shown(view, "delay")
    assert shown(view, "shape.phase")


class ErrorForm(FormModel):
    """A folded group holding a positive leaf and a grouped nested model."""

    gain: float = 1.0
    delay: Annotated[float, Display(group="Advanced")] = Field(default=1.0, gt=0)
    inner: Annotated[Section, Display(group="Advanced")] = Field(
        default_factory=Section
    )


def test_folded_group_title_shows_member_errors_without_unfolding(
    build: Build,
) -> None:
    view, state = build(ErrorForm, None)
    env = FormEnv()
    note = error_note(view, "Advanced")
    assert note.isHidden()

    state.write({("delay",): -1.0}, env)
    errors = view.errors()
    assert errors is not None
    assert set(errors.messages) == {("delay",)}
    assert not error_note(view, "Advanced").isHidden()
    assert not shown(view, "delay")
    assert group_fold(view, "Advanced").text() == "Expand"

    state.write({("delay",): 2.0}, env)
    assert view.errors() is None
    assert error_note(view, "Advanced").isHidden()

    # An error inside a grouped nested model marks the group too.
    state.write({("inner", "level"): 0.0}, env)
    assert not error_note(view, "Advanced").isHidden()
    state.write({("inner", "level"): 0.5}, env)
    assert error_note(view, "Advanced").isHidden()
