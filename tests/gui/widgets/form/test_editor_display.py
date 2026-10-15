"""Display hints and fixed values change which rows the editor draws, and how."""

from collections.abc import Callable, Iterator
from typing import Annotated, Literal, Self

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from PyQt6.QtGui import QAction
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QWidget,
)
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormInput,
    FormModel,
    FormState,
    LinRange,
)
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import mouse_click
from tests.gui.widgets.form._field_errors import field_error
from tests.gui.widgets.form._text_input import commit_text

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
    """Text of the row's own leading label, without a fold title's mark.

    A foldable model row's leading label is its fold title form.fold.<path>;
    another row's is the unnamed label beside its value label.
    """
    titles = view.findChildren(QLabel, f"form.fold.{path}")
    if titles:
        (title,) = titles
        mark, label = title.text().split(" ", 1)
        assert mark in ("▸", "▾"), title.text()
        return label
    row = child(view, QLabel, f"form.value.{path}").parentWidget()
    assert row is not None
    # An empty name finds only unnamed children: the name label.
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


class DriveRef(BaseModel):
    """Key branch of a tagged ref-or-inline selection."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ref"] = "ref"
    name: str = ""


class DriveInline(BaseModel):
    """Inline branch whose module is an ordinary nested model."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["inline"] = "inline"
    module: Section = Field(default_factory=Section)


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
    commit_text(delay, "2.34567")
    assert state.input.data["delay"] == 2.34567
    assert delay.text() == "2.34567"
    state.write({("delay",): 4.0}, FormEnv())
    assert delay.text() == "4.0"

    mode = view.findChild(QAction, "form.mode.delay")
    assert isinstance(mode, QAction)
    mode.trigger()
    commit_text(child(view, QLineEdit, "form.input.delay"), "1 / 3")
    assert child(view, QLabel, "form.value.delay").text() == "= 0.333"


def test_one_value_literal_is_read_only_text_without_modes(build: Build) -> None:
    class LockForm(FormModel):
        reps: Literal[1000] = 1000
        style: Literal["const"] = "const"
        nqz: Literal[1, 2] = 2

    view, state = build(LockForm, None)
    before = state.input.data
    for path, text in (("reps", "1000"), ("style", "const")):
        assert row_names(view, path) == [
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
    commit_text(child(view, QLineEdit, "form.input.shown"), "2.0")
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
        drive: Annotated[DriveRef | DriveInline, Field(discriminator="kind")]
        choice: Annotated[Alpha | Beta, Field(discriminator="kind")]

    view, state = build(
        TitleForm,
        {
            "drive": {"kind": "inline", "module": {"level": 1.0}},
            # Without a tag no option matches; the row keeps its title.
            "choice": {"x": 3.0},
        },
    )
    assert row_label(view, "ro_cfg") == "Ro cfg"
    assert row_label(view, "labelled") == "Readout"
    assert row_label(view, "drive") == "Drive"
    # The inline option holds only an ordinary model, which draws no row.
    assert row_names(view, "drive.module") == []
    assert row_label(view, "drive.module.level") == "Level"
    assert row_label(view, "choice") == "Choice"
    state.write({("choice", "kind"): "alpha"}, FormEnv())
    assert row_label(view, "choice") == "Choice"


class Body(BaseModel):
    """An ordinary model whose fields carry their own presentation."""

    model_config = ConfigDict(extra="forbid")

    level: Annotated[
        float, Display(label="Drive level", unit="V", description="Output level")
    ] = 1.0
    shape: Literal["flat", "ramp"] = "flat"
    width: Annotated[float, Display(group="Advanced")] = 2.0

    @model_validator(mode="after")
    def validate_cover(self) -> Self:
        """Leave a model-wide error at this model's own path."""
        if self.width < self.level:
            raise ValueError("Width must cover level")
        return self


class Packed(BaseModel):
    """A branch whose only field besides its tag is an ordinary model."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["packed"] = "packed"
    payload: Body = Field(default_factory=Body)


class Bare(BaseModel):
    """A branch with no fields besides its tag."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["bare"] = "bare"


class Nullable(BaseModel):
    """A branch whose only field is a nullable model."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["optional"] = "optional"
    payload: Body | None = None


class Nested(BaseModel):
    """A branch whose only field is itself a selection."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["nested"] = "nested"
    payload: Annotated[Packed | Bare, Field(discriminator="kind")] = Field(
        default_factory=Packed
    )


class Pair(BaseModel):
    """A branch with an ordinary model beside another field."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["pair"] = "pair"
    payload: Body = Field(default_factory=Body)
    note: float = 0.0


def test_an_option_holding_one_ordinary_model_draws_that_models_rows(
    build: Build,
) -> None:
    class FlattenForm(FormModel):
        choice: Annotated[Packed | Bare, Field(discriminator="kind")] = Field(
            default_factory=Packed
        )

    view, state = build(FlattenForm, None)

    # The lone model has no row of its own; its fields keep their full paths.
    assert row_names(view, "choice.payload") == []
    assert row_label(view, "choice.payload.level") == "Drive level"
    level_row = child(view, QLabel, "form.value.choice.payload.level").parentWidget()
    assert level_row is not None
    texts = [
        item.text()
        for item in level_row.findChildren(
            QLabel, "", Qt.FindChildOption.FindDirectChildrenOnly
        )
    ]
    assert "V" in texts
    assert "Output level" in level_row.toolTip()
    advanced = child(view, QLabel, "form.fold.group.choice.Advanced")
    mouse_click(advanced, advanced.rect().center())
    width = child(view, QLineEdit, "form.input.choice.payload.width")
    assert any(
        block.objectName().startswith("form.group.") and block.isAncestorOf(width)
        for block in view.findChildren(QWidget)
    )

    shape = child(view, QComboBox, "form.input.choice.payload.shape")
    shape.setCurrentIndex(shape.findText("ramp"))
    assert state.input.data["choice"] == {
        "kind": "packed",
        "payload": {"level": 1.0, "shape": "ramp", "width": 2.0},
    }

    # An error of the undrawn model shows at the nearest drawn row.
    state.write({("choice", "payload", "level"): 5.0}, FormEnv())
    assert "Width must cover level" in field_error(view, "choice")


def test_an_option_whose_lone_field_is_nullable_or_selectable_keeps_its_row(
    build: Build,
) -> None:
    class KeepForm(FormModel):
        optional: Annotated[Nullable | Bare, Field(discriminator="kind")] = Field(
            default_factory=Nullable
        )
        nested: Annotated[Nested | Bare, Field(discriminator="kind")] = Field(
            default_factory=Nested
        )
        pair: Annotated[Pair | Bare, Field(discriminator="kind")] = Field(
            default_factory=Pair
        )

    view, _state = build(KeepForm, None)

    assert row_label(view, "optional.payload") == "Payload"
    assert child(view, QCheckBox, "form.enabled.optional.payload") is not None
    assert row_label(view, "nested.payload") == "Payload"
    assert child(view, QComboBox, "form.input.nested.payload") is not None
    assert row_label(view, "pair.payload") == "Payload"
    assert child(view, QLineEdit, "form.input.pair.note") is not None
