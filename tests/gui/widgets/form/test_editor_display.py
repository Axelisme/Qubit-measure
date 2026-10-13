"""Display hints and fixed values change which rows the editor draws, and how."""

from collections.abc import Callable, Iterator
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from PyQt6.QtGui import QAction
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QWidget
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormInput,
    FormModel,
    FormState,
    LinRange,
    ModuleInput,
)
from zcu_tools.gui.widgets.form import FormEditor

type View = tuple[FormEditor[FormModel], FormState[FormModel]]
type Build = Callable[[type[FormModel], dict[str, JsonValue] | None], View]


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


def row_names(view: QWidget, path: str) -> list[str]:
    """Object names of every public control or label that belongs to one row."""
    return sorted(
        widget.objectName()
        for widget in view.findChildren(QWidget)
        if widget.objectName().endswith(f".{path}")
    )


def row_label(view: QWidget, path: str) -> str:
    """Text of the row's own leading label, found beside its value label."""
    row = child(view, QLabel, f"form.value.{path}").parentWidget()
    assert row is not None
    labels = row.findChildren(QLabel, "", Qt.FindChildOption.FindDirectChildrenOnly)
    return labels[0].text()


class Section(BaseModel):
    """An ordinary nested model; its class name is not a section title."""

    model_config = ConfigDict(extra="forbid")

    level: float = 1.0


class Alpha(BaseModel):
    """First branch of a form-local tagged union."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["alpha"] = "alpha"
    x: float = 1.0


class Beta(BaseModel):
    """Second branch of the same union."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["beta"] = "beta"


def test_decimals_round_the_shown_number_without_rounding_input(build: Build) -> None:
    class DecimalsForm(FormModel):
        delay: Annotated[float, Display(decimals=3)] = 1.23456789
        plain: float = 1.23456789

    view, state = build(DecimalsForm, None)
    delay = child(view, QLineEdit, "form.input.delay")
    assert delay.text() == "1.235"
    assert child(view, QLineEdit, "form.input.plain").text() == "1.23456789"
    assert state.input.data["delay"] == 1.23456789

    # Typed digits beyond decimals stay as typed and reach the state whole.
    delay.setText("2.34567")
    assert state.input.data["delay"] == 2.34567
    assert delay.text() == "2.34567"
    state.write({("delay",): 4.0}, FormEnv())
    assert delay.text() == "4.0"

    mode = view.findChild(QAction, "form.mode.delay")
    assert isinstance(mode, QAction)
    mode.trigger()
    child(view, QLineEdit, "form.input.delay").setText("1 / 3")
    assert child(view, QLabel, "form.value.delay").text() == "0.333"


def test_one_value_literal_is_read_only_text_without_modes(build: Build) -> None:
    class LockForm(FormModel):
        reps: Literal[1000] = 1000
        style: Literal["const"] = "const"
        nqz: Literal[1, 2] = 2

    view, state = build(LockForm, None)
    before = state.input.data
    for path, text in (("reps", "1000"), ("style", "const")):
        assert row_names(view, path) == [
            f"form.error.{path}",
            f"form.input.{path}",
            f"form.value.{path}",
        ]
        assert view.findChild(QAction, f"form.mode.{path}") is None
        fixed = child(view, QLineEdit, f"form.input.{path}")
        assert fixed.isReadOnly()
        assert fixed.text() == text
    assert state.input.data == before
    # Two legal values keep the dropdown, which has no Expression action.
    nqz = child(view, QComboBox, "form.input.nqz")
    assert [nqz.itemData(index) for index in range(nqz.count())] == [1, 2]
    assert view.findChild(QAction, "form.mode.nqz") is None


def test_hidden_field_is_not_drawn_but_kept_and_validated(build: Build) -> None:
    class HiddenForm(FormModel):
        shown: float = 1.0
        gain: Annotated[float, Display(hidden=True)] = Field(default=0.5, gt=0)

    view, state = build(HiddenForm, None)
    assert row_names(view, "gain") == []
    child(view, QLineEdit, "form.input.shown").setText("2.0")
    assert state.input.data == {"shown": 2.0, "gain": 0.5}
    assert view.errors() is None

    state.write({("gain",): -1.0}, FormEnv())
    errors = view.errors()
    assert errors is not None
    assert set(errors.messages) == {("gain",)}
    root = child(view, QLabel, "form.error")
    assert not root.isHidden()
    assert errors.messages["gain",] in root.text()


def test_one_value_tag_outside_a_union_is_not_drawn(build: Build) -> None:
    class TagForm(FormModel):
        choice: Annotated[Alpha | Beta, Field(discriminator="kind")] = Field(
            default_factory=Alpha
        )
        alone: Alpha = Field(default_factory=Alpha)
        sweep: LinRange = Field(default_factory=LinRange)

    view, _state = build(TagForm, None)
    assert row_names(view, "alone.kind") == []
    assert row_names(view, "sweep.kind") == []
    assert child(view, QLineEdit, "form.input.sweep.start").text() == "0.0"
    selector = child(view, QComboBox, "form.input.choice")
    assert [selector.itemData(i) for i in range(selector.count())] == [
        {"kind": "alpha"},
        {"kind": "beta"},
    ]
    assert row_names(view, "choice.kind") == []


def test_sections_and_selectors_are_titled_by_field_name(build: Build) -> None:
    class TitleForm(FormModel):
        ro_cfg: Section = Field(default_factory=Section)
        labelled: Annotated[Section, Display(label="Readout")] = Field(
            default_factory=Section
        )
        drive: ModuleInput
        choice: Annotated[Alpha | Beta, Field(discriminator="kind")]

    view, state = build(
        TitleForm,
        {
            "drive": {"kind": "inline", "module": {"type": "pulse"}},
            # Without a tag no option matches; the row keeps its title.
            "choice": {"x": 3.0},
        },
    )
    assert row_label(view, "ro_cfg") == "Ro cfg"
    assert row_label(view, "labelled") == "Readout"
    assert row_label(view, "drive") == "Drive"
    assert row_label(view, "drive.module") == "Module"
    assert row_label(view, "choice") == "Choice"
    state.write({("choice", "kind"): "alpha"}, FormEnv())
    assert row_label(view, "choice") == "Choice"
