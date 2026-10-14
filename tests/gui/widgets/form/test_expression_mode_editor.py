"""The right-click Expression action switches a text row between Direct and Expression."""

from collections.abc import Callable, Iterator, Mapping
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from PyQt6.QtGui import QAction
from qtpy.QtCore import QEvent, QPoint, Qt, QTimer
from qtpy.QtGui import QColor, QContextMenuEvent, QKeyEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QMenu,
    QWidget,
)
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    FormWritePath,
    WriteRejected,
)
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._form_context import captured_ext_env


class Alpha(BaseModel):
    """First branch of a tagged union; its tag is a structural selector."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["alpha"] = "alpha"


class Beta(BaseModel):
    """Second branch of the same union."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["beta"] = "beta"


class ModeForm(FormModel):
    """Every text row kind, which offers Expression mode, and rows that do not."""

    amount: float = 2.0
    count: int = 3
    amplitude: complex = 1 + 2j
    name: str = "literal"
    unit: Literal["native", "mA"] = "native"
    nqz: Literal[1, 2] = 2
    gain: float = 1.0
    enabled: bool = True
    fixed: Literal["const"] = "const"
    branch: Annotated[Alpha | Beta, Field(discriminator="kind")] = Field(
        default_factory=Alpha
    )

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Make gain a choice leaf of a float field."""
        return {("gain",): (1.0, 2.0)}


TEXT_ROWS = ("amount", "count", "amplitude", "name")
# Dropdown rows and their items' data, in order.
CHOICE_ROWS: dict[str, list[JsonValue]] = {
    "unit": ["native", "mA"],
    "nqz": [1, 2],
    "gain": [1.0, 2.0],
}


@pytest.fixture
def editor(
    qapp: QApplication,
) -> Iterator[tuple[FormEditor[ModeForm], FormState[ModeForm]]]:
    """Own one editor over default input; close and delete it afterwards."""
    env = FormEnv()
    state = FormState(ModeForm.default_input(env))
    view = FormEditor(state, env)
    yield view, state
    view.close()
    view.deleteLater()
    qapp.processEvents()


def child[W: QWidget](view: QWidget, kind: type[W], name: str) -> W:
    """Find a documented public child widget by its complete object name."""
    widget = view.findChild(kind, name)
    assert isinstance(widget, kind), name
    return widget


def mode(view: QWidget, path: str) -> QAction:
    """Find the documented Expression action of one row."""
    action = view.findChild(QAction, f"form.mode.{path}")
    assert isinstance(action, QAction), path
    return action


def row_combos(view: QWidget, path: str) -> list[str]:
    """Object names of the not-hidden combo boxes that belong to one row."""
    return sorted(
        combo.objectName()
        for combo in view.findChildren(QComboBox)
        if combo.objectName().endswith(f".{path}") and not combo.isHidden()
    )


def context_menu(widget: QWidget) -> list[QAction | None]:
    """Right-click widget and return the entries of the menu it opens.

    A separator is None. Returns [] when no menu opens. The menu is closed
    before returning, and its own standard actions may already be deleted.
    """
    shown: list[QAction | None] = []

    def record() -> None:
        menu = QApplication.activePopupWidget()
        if isinstance(menu, QMenu):
            shown.extend(
                None if item.isSeparator() else item for item in menu.actions()
            )
            menu.close()

    QTimer.singleShot(0, record)
    point = QPoint(4, widget.height() // 2)
    QApplication.sendEvent(
        widget,
        QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, point, widget.mapToGlobal(point)
        ),
    )
    QApplication.processEvents()
    return shown


def background(widget: QWidget) -> QColor:
    """Rendered colour near the right edge of an input, where no text sits."""
    image = widget.grab().toImage()
    return image.pixelColor(image.width() - 6, image.height() // 2)


def test_text_rows_list_the_expression_action_instead_of_a_mode_dropdown(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
) -> None:
    view, _state = editor
    for path in TEXT_ROWS:
        action = mode(view, path)
        assert action.isCheckable(), path
        assert not action.isChecked(), path
        assert view.findChild(QComboBox, f"form.mode.{path}") is None, path
        control = child(view, QWidget, f"form.input.{path}")
        assert action in control.actions(), path
        assert row_combos(view, path) == [], path
    # Dropdowns, checkboxes, a one-value Literal and a discriminator have none.
    for path in (*CHOICE_ROWS, "enabled", "fixed", "branch.kind"):
        assert view.findChild(QAction, f"form.mode.{path}") is None, path


@pytest.mark.parametrize(
    ("path", "text", "direct"),
    [
        ("amount", "2.0", 2.0),
        ("count", "3", 3),
        ("amplitude", "(1+2j)", {"__complex__": [1.0, 2.0]}),
        ("name", "literal", "literal"),
    ],
)
def test_expression_action_starts_from_the_direct_text_and_returns(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
    path: str,
    text: str,
    direct: JsonValue,
) -> None:
    view, state = editor
    action = mode(view, path)
    action.trigger()
    assert action.isChecked()
    assert state.input.data[path] == {"expr": text}
    assert child(view, QLineEdit, f"form.input.{path}").text() == text

    action.trigger()
    assert not action.isChecked()
    assert state.input.data[path] == direct
    assert child(view, QLineEdit, f"form.input.{path}").text() == text


def test_choice_dropdowns_list_only_the_choices_and_write_the_picked_one(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
) -> None:
    view, state = editor
    for path, options in CHOICE_ROWS.items():
        assert row_combos(view, path) == [f"form.input.{path}"], path
        direct = child(view, QComboBox, f"form.input.{path}")
        assert [direct.itemData(i) for i in range(direct.count())] == options
        assert direct.actions() == [], path
        direct.setCurrentIndex(0)
        assert state.input.data[path] == options[0], path


def test_expression_input_has_its_own_background(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
) -> None:
    view, _state = editor
    amount = child(view, QLineEdit, "form.input.amount")
    plain = background(child(view, QLineEdit, "form.input.count"))
    assert background(amount) == plain
    mode(view, "amount").trigger()
    assert background(amount) != plain
    mode(view, "amount").trigger()
    assert background(amount) == plain


def test_text_menu_keeps_the_standard_items_then_offers_the_expression_action(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
) -> None:
    view, _state = editor
    action = mode(view, "amount")
    shown = context_menu(child(view, QLineEdit, "form.input.amount"))
    assert shown[-1] is action
    assert shown[-2] is None
    # The standard text-editing items come first.
    assert len([item for item in shown[:-2] if item is not None]) >= 2


def test_dropdown_and_checkbox_rows_open_no_expression_menu(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
) -> None:
    view, _state = editor
    for path in CHOICE_ROWS:
        assert context_menu(child(view, QComboBox, f"form.input.{path}")) == [], path
    assert context_menu(child(view, QCheckBox, "form.input.enabled")) == []


class FillForm(FormModel):
    """Text rows whose computed Expression values fill Direct mode."""

    amount: float = Field(default=2.0, gt=0)
    count: int = 3
    amplitude: complex = 1 + 0j
    delay: Annotated[float, Display(decimals=3)] = 1.0


type Batch = dict[FormWritePath, JsonValue]
type Fill = tuple[FormEditor[FillForm], FormState[FillForm], list[Batch]]
type BuildFill = Callable[[FormEnv, Callable[[Batch], None] | None], Fill]


@pytest.fixture
def fill(qapp: QApplication) -> Iterator[BuildFill]:
    """Build editors over FillForm whose host write records accepted batches.

    The optional check runs before a batch is recorded and installed and may
    raise WriteRejected to refuse it.
    """
    editors: list[FormEditor[FillForm]] = []

    def make(env: FormEnv, check: Callable[[Batch], None] | None) -> Fill:
        state = FormState(FormInput(FillForm, {}))
        state.write(
            {
                ("amount",): 2.0,
                ("count",): 3,
                ("amplitude",): {"__complex__": [1.0, 0.0]},
                ("delay",): 1.0,
            },
            env,
        )
        batches: list[Batch] = []

        def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
            if check is not None:
                check(dict(batch))
            batches.append(dict(batch))
            state.write(batch, env)

        def run_command(path: FormWritePath, name: str) -> None:
            state.run_command(path, name, env)

        view = FormEditor(state, env, write=write, run_command=run_command)
        editors.append(view)
        return view, state, batches

    yield make
    for view in editors:
        view.close()
        view.deleteLater()
    qapp.processEvents()


def seed_env() -> FormEnv:
    """A fixed capture whose general.ext.seed is 4321.5."""
    return captured_ext_env(seed=4321.5)


def type_text(edit: QLineEdit, text: str) -> None:
    """Replace edit's whole text by typing text key by key, without Enter."""
    edit.selectAll()
    for char in text:
        code = Qt.Key(ord(char.upper()))
        for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            event = QKeyEvent(kind, code, Qt.KeyboardModifier.NoModifier, char)
            QApplication.sendEvent(edit, event)


@pytest.mark.parametrize(
    ("path", "expression", "direct", "text"),
    [
        ("amount", "general.ext.seed + 5", 4326.5, "4326.5"),
        ("count", "2 * 3", 6, "6"),
        ("amplitude", "1j * 2 + 1", {"__complex__": [1.0, 2.0]}, "(1+2j)"),
    ],
)
def test_direct_mode_fills_the_computed_expression_value(
    fill: BuildFill, path: str, expression: str, direct: JsonValue, text: str
) -> None:
    view, state, batches = fill(seed_env(), None)
    state.write({(path,): {"expr": expression}}, seed_env())
    action = mode(view, path)
    assert action.isChecked()

    action.trigger()

    assert not action.isChecked()
    assert batches == [{(path,): direct}]
    assert state.input.data[path] == direct
    assert child(view, QLineEdit, f"form.input.{path}").text() == text
    assert view.errors() is None


def test_direct_mode_keeps_every_digit_of_a_rounded_value(fill: BuildFill) -> None:
    view, state, batches = fill(seed_env(), None)
    state.write({("delay",): {"expr": "1 / 3"}}, seed_env())

    mode(view, "delay").trigger()

    assert batches == [{("delay",): 1 / 3}]
    assert state.input.data["delay"] == 1 / 3
    # Only the shown text is rounded to the row's three decimals.
    assert child(view, QLineEdit, "form.input.delay").text() == "0.333"


@pytest.mark.parametrize(
    ("path", "expression", "direct", "message"),
    [
        ("amount", "0 - 2", -2, "greater than 0"),
        ("count", "1 / 2", 0.5, "valid integer"),
    ],
)
def test_direct_mode_fills_a_computed_value_the_model_rejects(
    fill: BuildFill, path: str, expression: str, direct: JsonValue, message: str
) -> None:
    view, state, batches = fill(seed_env(), None)
    state.write({(path,): {"expr": expression}}, seed_env())

    mode(view, path).trigger()

    assert batches == [{(path,): direct}]
    assert state.input.data[path] == direct
    assert view.local_parse_errors() is None
    error = child(view, QLabel, f"form.error.{path}")
    assert message in error.text()
    assert not error.isHidden()


@pytest.mark.parametrize(
    ("env", "expression"),
    [
        (seed_env(), "1 +"),
        (seed_env(), "missing_name"),
        (seed_env(), "1 / 0"),
        (FormEnv(), "general.ext.seed + 5"),
    ],
    ids=["syntax", "unknown-reference", "arithmetic", "no-context"],
)
def test_direct_mode_keeps_an_expression_that_does_not_compute(
    fill: BuildFill, env: FormEnv, expression: str
) -> None:
    view, state, batches = fill(env, None)
    state.write({("amount",): {"expr": expression}}, env)
    action = mode(view, "amount")

    action.trigger()

    assert not action.isChecked()
    assert child(view, QLineEdit, "form.input.amount").text() == expression
    assert batches == []
    assert state.input.data["amount"] == {"expr": expression}
    local = view.local_parse_errors()
    assert local is not None
    assert set(local.messages) == {("amount",)}


def test_direct_mode_ignores_the_value_of_uncommitted_expression_text(
    fill: BuildFill,
) -> None:
    view, state, batches = fill(seed_env(), None)
    state.write({("amount",): {"expr": "1 / 4"}}, seed_env())
    amount = child(view, QLineEdit, "form.input.amount")

    type_text(amount, "1 / 8")
    batches.clear()
    mode(view, "amount").trigger()

    assert amount.text() == "1 / 8"
    assert state.input.data["amount"] != 0.25
    assert all(0.25 not in batch.values() for batch in batches)
    local = view.local_parse_errors()
    assert local is not None
    assert set(local.messages) == {("amount",)}


def test_direct_mode_ignores_the_value_of_rejected_expression_text(
    fill: BuildFill,
) -> None:
    def reject_doubling(batch: Batch) -> None:
        if batch.get(("amount",)) == {"expr": "2 * 4"}:
            raise WriteRejected("Not allowed here")

    view, state, batches = fill(seed_env(), reject_doubling)
    state.write({("amount",): {"expr": "1 / 4"}}, seed_env())
    amount = child(view, QLineEdit, "form.input.amount")
    amount.setText("2 * 4")
    for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
        event = QKeyEvent(kind, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(amount, event)
    batches.clear()

    mode(view, "amount").trigger()

    assert amount.text() == "2 * 4"
    assert batches == []
    assert state.input.data["amount"] == {"expr": "1 / 4"}


def test_read_only_rows_do_not_switch_to_direct(fill: BuildFill) -> None:
    view, state, batches = fill(seed_env(), None)
    state.write({("amount",): {"expr": "general.ext.seed + 5"}}, seed_env())
    view.set_editing_enabled(enabled=False)
    action = mode(view, "amount")

    action.trigger()

    assert action.isChecked()
    assert batches == []
    assert state.input.data["amount"] == {"expr": "general.ext.seed + 5"}
