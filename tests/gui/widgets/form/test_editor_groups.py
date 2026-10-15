"""Display groups fold a model level's grouped rows into one block per group.

Group blocks and model sections fold by a click on their bold title.
"""

from collections.abc import Callable, Iterator
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QWidget,
)
from zcu_tools.gui.form import Display, FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import key_click, mouse_click, type_text

type View = tuple[FormEditor[FormModel], FormState[FormModel]]
type Build = Callable[[type[FormModel], dict[str, JsonValue] | None], View]

ERROR_NOTE = "Contains errors"
# A fold title starts with one of these, then the section's label.
FOLDED, EXPANDED = "\u25b8 ", "\u25be "


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
    """Children of kind whose parent widget is parent itself, named or not."""
    return [widget for widget in parent.children() if isinstance(widget, kind)]


def group_fold(view: QWidget, name: str) -> QLabel:
    """The fold title of the group block form.group.<name>."""
    title = child(view, QLabel, f"form.fold.group.{name}")
    assert child(view, QWidget, f"form.group.{name}").isAncestorOf(title), name
    return title


def model_fold(view: QWidget, path: str) -> QLabel:
    """The fold title in the header of the model row at path."""
    title = child(view, QLabel, f"form.fold.{path}")
    assert title.parentWidget() is row_of(view, path), path
    return title


def fold(title: QLabel) -> None:
    """Click title, as a user folds or unfolds its section."""
    mouse_click(title, title.rect().center())


def expanded(title: QLabel) -> bool:
    """Whether title's mark says its section is expanded."""
    text = title.text()
    assert text.startswith((FOLDED, EXPANDED)), text
    return text.startswith(EXPANDED)


def label_of(view: QWidget, path: str, text: str) -> QLabel:
    """The name label showing text in the header of the row at path."""
    (label,) = [
        label for label in direct(row_of(view, path), QLabel) if label.text() == text
    ]
    return label


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
    field = view.findChild(QWidget, f"form.input.{path}")
    return field is not None and field.isVisibleTo(view)


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
    for name in members:
        assert group_fold(view, name).text() == FOLDED + name.split(".")[-1], name
        assert not expanded(group_fold(view, name)), name
    for path in ("mode", "after", "drive.ch", "drive.freq"):
        assert shown(view, path), path
        assert not child(view, QWidget, "form.group.Advanced").isAncestorOf(
            child(view, QWidget, f"form.input.{path}")
        ), path
    # A group whose every member is hidden has no block.
    assert view.findChild(QWidget, "form.group.Hidden") is None
    assert view.findChildren(QWidget, "form.input.kept") == []

    for name in members:
        fold(group_fold(view, name))
    view.resize(600, view.sizeHint().height())
    view.show()
    qapp.processEvents()
    for name, paths in members.items():
        assert group_fold(view, name).text() == EXPANDED + name.split(".")[-1], name
        block = child(view, QWidget, f"form.group.{name}")
        assert all(block.isAncestorOf(row_of(view, path)) for path in paths), name
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
    # A group title sits in its level's label column like a row's name label:
    # same left edge and width, so the level's controls still share one edge.
    title = group_fold(view, "Advanced")
    mode_label = label_of(view, "mode", "Mode")
    assert (
        title.mapTo(view, QPoint(0, 0)).x() == mode_label.mapTo(view, QPoint(0, 0)).x()
    )
    assert title.width() == mode_label.width()
    mode_left = child(view, QComboBox, "form.input.mode").mapTo(view, QPoint(0, 0))
    assert mode_left.x() > title.mapTo(view, QPoint(0, 0)).x() + title.width()
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
    selector = child(view, QComboBox, "form.input.shape")
    selector.setCurrentIndex(selector.findText(tag))


def test_user_folds_survive_a_rebuild_of_the_same_editor(build: Build) -> None:
    view, state = build(FoldForm, None)
    assert not shown(view, "delay")
    assert not shown(view, "shape.phase")
    assert shown(view, "drive.level")
    assert not shown(view, "extra.level")

    fold(group_fold(view, "Advanced"))
    fold(group_fold(view, "shape.Advanced"))
    fold(model_fold(view, "drive"))
    fold(model_fold(view, "extra"))
    switch_shape(view, "beta")

    shape = state.input.data["shape"]
    assert isinstance(shape, dict)
    assert shape["kind"] == "beta"
    assert view.findChild(QLineEdit, "form.input.shape.y") is not None
    assert shown(view, "delay")
    assert shown(view, "shape.phase")
    assert expanded(group_fold(view, "Advanced"))
    assert not shown(view, "drive.level")
    assert not expanded(model_fold(view, "drive"))
    assert expanded(model_fold(view, "extra"))
    assert shown(view, "extra.level")
    # Untouched groups keep their folded default.
    assert not shown(view, "window")

    # Folding back is remembered as well.
    fold(group_fold(view, "Advanced"))
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
    assert not expanded(group_fold(view, "Advanced"))

    state.write({("delay",): 2.0}, env)
    assert view.errors() is None
    assert error_note(view, "Advanced").isHidden()

    # An error inside a grouped nested model marks the group too.
    state.write({("inner", "level"): 0.0}, env)
    assert not error_note(view, "Advanced").isHidden()
    state.write({("inner", "level"): 0.5}, env)
    assert error_note(view, "Advanced").isHidden()


class GroupFirst(FormModel):
    """A grouped row declared before the ungrouped text row it follows on screen."""

    phase: Annotated[float, Display(group="Advanced")] = 0.0
    amount: float = 1.0


class GroupLast(FormModel):
    """The same rows with the grouped row declared last."""

    amount: float = 1.0
    phase: Annotated[float, Display(group="Advanced")] = 0.0


class InnerGroupFirst(BaseModel):
    """A nested level whose grouped row is declared before its ungrouped row."""

    model_config = ConfigDict(extra="forbid")

    phase: Annotated[float, Display(group="Advanced")] = 0.0
    amount: float = 1.0


class NestedGroupFirst(FormModel):
    """A root holding one nested level with a group declared first."""

    drive: InnerGroupFirst = Field(default_factory=InnerGroupFirst)


def value_at(data: dict[str, JsonValue], path: str) -> JsonValue:
    """The input value at the dotted path inside data."""
    value: JsonValue = data
    for key in path.split("."):
        assert isinstance(value, dict), path
        value = value[key]
    return value


@pytest.mark.parametrize(
    ("model", "path", "group"),
    [
        (GroupFirst, "amount", None),
        (GroupLast, "amount", None),
        (GroupFirst, "phase", "Advanced"),
        (NestedGroupFirst, "drive.amount", None),
        (NestedGroupFirst, "drive.phase", "drive.Advanced"),
    ],
)
def test_typing_keeps_the_text_field_whatever_the_declaration_order(
    build: Build,
    qapp: QApplication,
    model: type[FormModel],
    path: str,
    group: str | None,
) -> None:
    view, state = build(model, None)
    if group is not None:
        fold(group_fold(view, group))
    view.show()
    qapp.processEvents()
    field = child(view, QLineEdit, f"form.input.{path}")
    field.setFocus()
    qapp.processEvents()

    typed = field.text() + "55"
    key_click(field, Qt.Key.Key_End)
    type_text(field, "55")
    key_click(field, Qt.Key.Key_Return)
    qapp.processEvents()

    # Enter writes the state; a write that keeps the form's shape updates
    # the row in place, so the field the user types into stays and keeps focus.
    assert child(view, QLineEdit, f"form.input.{path}") is field
    assert field.hasFocus()
    assert field.text() == typed
    assert value_at(state.input.data, path) == float(typed)


class Extra(BaseModel):
    """The model of a nullable section."""

    model_config = ConfigDict(extra="forbid")

    note: str = "n"


class Muted(BaseModel):
    """A tone branch without editable fields."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["muted"] = "muted"


class Tone(BaseModel):
    """A tone branch whose option draws one field."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tone"] = "tone"
    gain: float = 0.5


class ShapeForm(FormModel):
    """A group declared first, then every kind of row that changes the shape."""

    delay: Annotated[float, Display(group="Advanced")] = 0.0
    shape: Annotated[Alpha | Beta, Field(discriminator="kind")] = Field(
        default_factory=Alpha
    )
    extra: Extra | None = None
    tone: Annotated[Muted | Tone, Field(discriminator="kind")] = Field(
        default_factory=Muted
    )


def test_shape_changes_rebuild_rows_and_keep_folds_with_groups_first(
    build: Build,
) -> None:
    view, state = build(ShapeForm, None)
    fold(group_fold(view, "Advanced"))
    fold(group_fold(view, "shape.Advanced"))

    def folds_kept() -> bool:
        return (
            shown(view, "delay")
            and shown(view, "shape.phase")
            and expanded(group_fold(view, "Advanced"))
        )

    switch_shape(view, "beta")
    assert view.findChild(QLineEdit, "form.input.shape.y") is not None
    assert view.findChild(QLineEdit, "form.input.shape.x") is None
    assert folds_kept()

    assert view.findChild(QLineEdit, "form.input.extra.note") is None
    child(view, QCheckBox, "form.enabled.extra").setChecked(True)
    assert state.input.data["extra"] is not None
    assert view.findChild(QLineEdit, "form.input.extra.note") is not None
    assert folds_kept()

    assert view.findChild(QLineEdit, "form.input.tone.gain") is None
    tone = child(view, QComboBox, "form.input.tone")
    tone.setCurrentIndex(tone.findText("tone"))
    assert state.input.data["tone"] == {"kind": "tone", "gain": 0.5}
    assert view.findChild(QLineEdit, "form.input.tone.gain") is not None
    assert folds_kept()


class Packed(BaseModel):
    """A branch whose only field besides its tag is an ordinary model."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["packed"] = "packed"
    payload: Section = Field(default_factory=Section)


class Bare(BaseModel):
    """A branch with no fields besides its tag."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["bare"] = "bare"


class SectionsForm(FormModel):
    """Every foldable section beside plain rows.

    Foldable: a nested model, a tagged branch, an option holding one ordinary
    model, an enabled nullable model and a group. Plain: a number, a choice
    and a disabled nullable model.
    """

    gain: float = 1.0
    mode: Literal["fast", "slow"] = "fast"
    drive: Section = Field(default_factory=Section)
    shape: Annotated[Alpha | Beta, Field(discriminator="kind")] = Field(
        default_factory=Alpha
    )
    wrapped: Annotated[Packed | Bare, Field(discriminator="kind")] = Field(
        default_factory=Packed
    )
    extra: Extra | None = None
    off: Extra | None = None
    delay: Annotated[float, Display(group="Advanced")] = 0.0


SECTIONS: dict[str, JsonValue] = {
    "gain": 1.0,
    "mode": "fast",
    "drive": {"level": 1.0},
    "shape": {"kind": "alpha", "x": 1.0, "phase": 0.0},
    "wrapped": {"kind": "packed", "payload": {"level": 1.0}},
    "extra": {"note": "n"},
    "off": None,
    "delay": 0.0,
}


def test_every_section_folds_by_a_click_on_its_bold_title(build: Build) -> None:
    view, state = build(SectionsForm, SECTIONS)
    for title, label, inner in (
        (model_fold(view, "drive"), "Drive", "drive.level"),
        (model_fold(view, "shape"), "Shape", "shape.x"),
        (model_fold(view, "wrapped"), "Wrapped", "wrapped.payload.level"),
        (model_fold(view, "extra"), "Extra", "extra.note"),
        (group_fold(view, "Advanced"), "Advanced", "delay"),
    ):
        start = expanded(title)
        assert title.font().bold(), label
        assert title.text() == (EXPANDED if start else FOLDED) + label
        assert shown(view, inner) == start, inner
        fold(title)
        assert title.text() == (FOLDED if start else EXPANDED) + label
        assert shown(view, inner) != start, inner
        fold(title)
        assert title.text() == (EXPANDED if start else FOLDED) + label
        assert shown(view, inner) == start, inner
    assert state.input.data == SECTIONS


def test_plain_rows_keep_a_normal_weight_name_without_a_mark(build: Build) -> None:
    view, state = build(SectionsForm, SECTIONS)
    for path in ("drive", "shape", "wrapped", "extra"):
        title = model_fold(view, path)
        assert title.font().bold() and expanded(title), path
    plain = {"gain": "Gain", "mode": "Mode", "off": "Off", "drive.level": "Level"}
    for path, text in plain.items():
        assert not label_of(view, path, text).font().bold(), path
    # A disabled nullable model has nothing to fold: its name ignores a click.
    off = label_of(view, "off", "Off")
    mouse_click(off, off.rect().center())
    assert label_of(view, "off", "Off") is off
    assert not child(view, QCheckBox, "form.enabled.off").isChecked()
    assert state.input.data == SECTIONS


def tab(start: QWidget, until: Callable[[QWidget], bool]) -> list[str]:
    """Focus start, then press Tab on the focused widget until until(focused).

    Stops when focus comes back to start, raising if until never held.
    Returns the object name of each widget focus stopped on, start first.
    """
    start.setFocus(Qt.FocusReason.TabFocusReason)
    assert start.hasFocus(), start.objectName()
    stops = [start.objectName()]
    for _ in range(100):
        focused = QApplication.focusWidget()
        assert focused is not None
        key_click(focused, Qt.Key.Key_Tab)
        now = QApplication.focusWidget()
        assert now is not None
        if until(now):
            return stops
        if now is start:
            raise AssertionError(f"Tab came back to the start: {stops}")
        stops.append(now.objectName())
    raise AssertionError(f"Tab did not come back to the start: {stops}")


def tab_to(view: QWidget, name: str) -> QLabel:
    """Press Tab from the first input until the fold title name has focus."""
    title = child(view, QLabel, name)
    tab(child(view, QWidget, "form.input.gain"), lambda now: now is title)
    return title


def press(key: Qt.Key, text: str = "") -> None:
    """Press and release key on the widget that has keyboard focus."""
    focused = QApplication.focusWidget()
    assert focused is not None
    key_click(focused, key, text=text)


SECTION_TITLES = (
    ("form.fold.drive", "drive.level"),
    ("form.fold.shape", "shape.x"),
    ("form.fold.wrapped", "wrapped.payload.level"),
    ("form.fold.extra", "extra.note"),
    ("form.fold.group.Advanced", "delay"),
)


def test_tab_stops_on_fold_titles_and_space_or_enter_folds_them(
    build: Build, qapp: QApplication
) -> None:
    view, state = build(SectionsForm, SECTIONS)
    view.resize(600, 700)
    view.show()
    qapp.processEvents()
    start = child(view, QWidget, "form.input.gain")
    stops = tab(start, lambda now: now is start)
    for name, _inner in SECTION_TITLES:
        assert name in stops, (name, stops)
    for name, inner in SECTION_TITLES:
        title = tab_to(view, name)
        start_expanded = expanded(title)
        for key, text in (
            (Qt.Key.Key_Space, " "),
            (Qt.Key.Key_Return, "\r"),
            (Qt.Key.Key_Enter, "\r"),
            (Qt.Key.Key_Space, " "),
        ):
            before = expanded(title)
            press(key, text)
            assert expanded(title) != before, (name, key)
            assert shown(view, inner) != before, (name, key)
        assert expanded(title) == start_expanded, name
        # No other key folds a title.
        for key in (Qt.Key.Key_Right, Qt.Key.Key_Left, Qt.Key.Key_Down):
            press(key)
            assert expanded(title) == start_expanded, (name, key)
    assert state.input.data == SECTIONS
    # Enter in a text field inside a section commits the text, not a fold.
    level = child(view, QLineEdit, "form.input.drive.level")
    level.setFocus(Qt.FocusReason.TabFocusReason)
    key_click(level, Qt.Key.Key_End)
    type_text(level, "5")
    key_click(level, Qt.Key.Key_Return)
    assert state.input.data["drive"] == {"level": 1.05}
    assert expanded(model_fold(view, "drive"))
    assert shown(view, "drive.level")


def edges_changed(before: QImage, after: QImage) -> list[str]:
    """The sides of two same-size images whose outermost pixel line differs."""
    assert before.size() == after.size()
    width, height = before.width(), before.height()
    sides = {
        "top": [(x, 0) for x in range(width)],
        "bottom": [(x, height - 1) for x in range(width)],
        "left": [(0, y) for y in range(height)],
        "right": [(width - 1, y) for y in range(height)],
    }
    return [
        side
        for side, points in sides.items()
        if any(before.pixel(x, y) != after.pixel(x, y) for x, y in points)
    ]


def test_a_fold_title_reached_by_tab_shows_an_outline(
    build: Build, qapp: QApplication
) -> None:
    view, _state = build(SectionsForm, SECTIONS)
    view.resize(600, 700)
    view.show()
    qapp.processEvents()
    for name, _inner in SECTION_TITLES:
        title = child(view, QLabel, name)
        unfocused = title.grab().toImage()
        tab_to(view, name)
        focused = title.grab().toImage()
        # A frame around the title: every side of its rendering changes.
        assert edges_changed(unfocused, focused) == [
            "top",
            "bottom",
            "left",
            "right",
        ], name
        press(Qt.Key.Key_Tab)
        assert not title.hasFocus(), name
        assert title.grab().toImage() == unfocused, name


class Span(BaseModel):
    """A frequency span with two leaves and no unit of its own."""

    model_config = ConfigDict(extra="forbid")

    start: float = 4000.0
    stop: float = 4100.0


class SweepForm(FormModel):
    """One model section whose Display carries a unit."""

    freq: Annotated[Span, Display(label="Freq", unit="MHz")] = Field(
        default_factory=Span
    )


def test_a_section_unit_sits_right_after_its_name(
    build: Build, qapp: QApplication
) -> None:
    view, _state = build(SweepForm, None)
    view.resize(600, view.sizeHint().height())
    view.show()
    qapp.processEvents()
    row = row_of(view, "freq")
    (name,) = [label for label in direct(row, QLabel) if label.text().endswith("Freq")]
    unit = label_of(view, "freq", "MHz")
    name_right = name.geometry().right()
    unit_left = unit.geometry().left()
    assert unit_left > name_right
    # Nothing of the row's header sits between the name and its unit.
    between = [
        type(widget).__name__
        for widget in direct(row, QWidget)
        if widget.isVisible()
        and widget.geometry().left() > name_right
        and widget.geometry().right() < unit_left
    ]
    assert between == []
