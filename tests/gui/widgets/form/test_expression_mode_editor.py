"""The right-click Expression action switches a row between Direct and Expression."""

from collections.abc import Iterator
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from PyQt6.QtGui import QAction
from qtpy.QtCore import QPoint, QTimer
from qtpy.QtGui import QColor, QContextMenuEvent
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit, QMenu, QWidget
from zcu_tools.gui.form import FormEnv, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor


class Alpha(BaseModel):
    """First branch of a tagged union; its tag is a structural selector."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["alpha"] = "alpha"


class Beta(BaseModel):
    """Second branch of the same union."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["beta"] = "beta"


class ModeForm(FormModel):
    """Every row kind that offers Expression mode, plus two that do not."""

    amount: float = 2.0
    count: int = 3
    amplitude: complex = 1 + 2j
    name: str = "literal"
    unit: Literal["native", "mA"] = "native"
    fixed: Literal["const"] = "const"
    branch: Annotated[Alpha | Beta, Field(discriminator="kind")] = Field(
        default_factory=Alpha
    )


TEXT_ROWS = ("amount", "count", "amplitude", "name")
SWITCHABLE = (*TEXT_ROWS, "unit")


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


def test_switchable_rows_list_the_expression_action_instead_of_a_mode_dropdown(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
) -> None:
    view, _state = editor
    for path in SWITCHABLE:
        action = mode(view, path)
        assert action.isCheckable(), path
        assert not action.isChecked(), path
        assert view.findChild(QComboBox, f"form.mode.{path}") is None, path
        control = child(view, QWidget, f"form.input.{path}")
        assert action in control.actions(), path
    for path in TEXT_ROWS:
        assert row_combos(view, path) == [], path
    assert row_combos(view, "unit") == ["form.input.unit"]
    # A one-value Literal and a discriminator have no Expression mode.
    assert view.findChild(QAction, "form.mode.fixed") is None
    assert view.findChild(QAction, "form.mode.branch.kind") is None


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


def test_choice_dropdown_lists_only_choices_and_switches_through_the_action(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
) -> None:
    view, state = editor
    direct = child(view, QComboBox, "form.input.unit")
    assert [direct.itemData(index) for index in range(direct.count())] == [
        "native",
        "mA",
    ]
    direct.setCurrentIndex(direct.findData("mA"))
    action = mode(view, "unit")
    action.trigger()
    assert state.input.data["unit"] == {"expr": "mA"}
    text = child(view, QLineEdit, "form.input.unit")
    assert text.text() == "mA"
    assert row_combos(view, "unit") == []
    assert action in text.actions()

    # An expression that does not resolve to a choice returns to the last one.
    text.setText("absent_name")
    action.trigger()
    assert state.input.data["unit"] == "mA"
    shown = child(view, QComboBox, "form.input.unit")
    assert shown.currentData() == "mA"
    assert shown.count() == 2
    assert action in shown.actions()


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


def test_choice_menu_offers_the_expression_action(
    editor: tuple[FormEditor[ModeForm], FormState[ModeForm]],
) -> None:
    view, _state = editor
    shown = context_menu(child(view, QComboBox, "form.input.unit"))
    assert shown == [mode(view, "unit")]
