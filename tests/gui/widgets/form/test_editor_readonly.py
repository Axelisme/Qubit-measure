"""set_editing_enabled(enabled=False) leaves the form readable but not editable.

Every check is a user action on a shown editor: typing, clicking, the mouse
wheel, arrow keys, the Expression action and the fold buttons. Each action is
also shown to edit the form once editing is enabled again, so a check of
"nothing changed" cannot pass because the action never reached the control.
"""

from collections.abc import Iterator, Mapping
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from PyQt6.QtCore import QPoint, QPointF, Qt, QTimer
from PyQt6.QtGui import QAction, QContextMenuEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QWidget,
)
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
)
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import key_click, mouse_click, type_text


class Section(BaseModel):
    """A nested model that starts folded."""

    model_config = ConfigDict(extra="forbid")

    level: float = 1.0


class Extra(BaseModel):
    """The model of a nullable section with an Enabled checkbox."""

    model_config = ConfigDict(extra="forbid")

    note: str = "n"


DRIVE_OPTIONS = (
    Option("Q1.pulses.pi", "Q1.pulses.pi"),
    Option("Q1.pulses.pi2", "Q1.pulses.pi2"),
)


class ReadOnlyForm(FormModel):
    """One row of every control kind the editor draws."""

    amount: float = 2.0
    label: str = "probe"
    mode: Literal["fast", "slow"] = "fast"
    flag: bool = False
    drive: str = "Q1.pulses.pi"
    section: Annotated[Section, Display(collapsed=True)] = Field(
        default_factory=Section
    )
    extra: Extra | None = None
    phase: Annotated[float, Display(group="Advanced")] = 0.5

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        """Offer two declared drive keys in one option dropdown."""
        return {("drive",): DRIVE_OPTIONS}


class GroupFirstForm(FormModel):
    """ReadOnlyForm's rows with the grouped row declared first; the editor
    still draws it in a group block after the ungrouped rows."""

    phase: Annotated[float, Display(group="Advanced")] = 0.5
    amount: float = 2.0
    label: str = "probe"
    mode: Literal["fast", "slow"] = "fast"
    flag: bool = False
    drive: str = "Q1.pulses.pi"
    section: Annotated[Section, Display(collapsed=True)] = Field(
        default_factory=Section
    )
    extra: Extra | None = None

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
        """Offer two declared drive keys in one option dropdown."""
        return {("drive",): DRIVE_OPTIONS}


DATA: dict[str, JsonValue] = {
    "amount": 2.0,
    "label": "probe",
    "mode": "fast",
    "flag": False,
    "drive": "Q1.pulses.pi",
    "section": {"level": 1.0},
    "extra": None,
    "phase": 0.5,
}
TEXT_ROWS = ("amount", "label")
DROPDOWN_ROWS = ("mode", "drive")


type View = tuple[FormEditor[FormModel], FormState[FormModel]]


@pytest.fixture(
    params=[ReadOnlyForm, GroupFirstForm], ids=["group-last", "group-first"]
)
def view(qapp: QApplication, request: pytest.FixtureRequest) -> Iterator[View]:
    """A shown editor over DATA, with editing disabled, for each field order."""
    model: type[FormModel] = request.param
    state = FormState(FormInput(model, DATA))
    editor = FormEditor(state, FormEnv())
    editor.resize(520, 600)
    editor.show()
    editor.set_editing_enabled(enabled=False)
    qapp.processEvents()
    yield editor, state
    editor.close()
    editor.deleteLater()
    qapp.processEvents()


def child[W: QWidget](view: QWidget, kind: type[W], name: str) -> W:
    """Find a documented public child by its complete object name."""
    widget = view.findChild(kind, name)
    assert isinstance(widget, kind), name
    return widget


def mode(view: QWidget, path: str) -> QAction:
    """The documented Expression action of one text row."""
    action = view.findChild(QAction, f"form.mode.{path}")
    assert isinstance(action, QAction), path
    return action


def enabled_box(view: QWidget, path: str) -> QCheckBox:
    """The Enabled checkbox of the nullable model row at path."""
    return child(view, QCheckBox, f"form.enabled.{path}")


def fold(block: QWidget) -> QPushButton:
    """The Expand/Collapse button directly inside a row or group block."""
    (button,) = [
        button
        for button in block.findChildren(
            QPushButton, "", Qt.FindChildOption.FindDirectChildrenOnly
        )
        if button.text() in ("Expand", "Collapse")
    ]
    return button


def row_of(view: QWidget, path: str) -> QWidget:
    """The row widget at path: the parent of its form.value.<path> label."""
    row = child(view, QLabel, f"form.value.{path}").parentWidget()
    assert row is not None
    return row


def wheel(widget: QWidget) -> None:
    """Turn the mouse wheel one notch down over widget's centre."""
    centre = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(
        centre,
        QPointF(widget.mapToGlobal(centre.toPoint())),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def click(widget: QWidget) -> None:
    """Left-click widget near its left edge, where a checkbox has its box."""
    mouse_click(widget, QPoint(6, widget.height() // 2))


def opens_popup(widget: QComboBox) -> bool:
    """Click the dropdown; return whether its list opened, and close it."""
    click(widget)
    opened = QApplication.activePopupWidget() is not None
    widget.hidePopup()
    QApplication.processEvents()
    return opened


def menu_entries(widget: QWidget) -> list[tuple[str, bool]]:
    """Right-click widget; return each non-separator menu entry's text and
    whether it is enabled, without shortcut text or mnemonics."""
    shown: list[tuple[str, bool]] = []

    def record() -> None:
        menu = QApplication.activePopupWidget()
        if isinstance(menu, QMenu):
            shown.extend(
                (item.text().split("\t")[0].replace("&", ""), item.isEnabled())
                for item in menu.actions()
                if not item.isSeparator()
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


def test_text_fields_ignore_typing_but_select_and_copy(
    view: View, qapp: QApplication
) -> None:
    editor, state = view
    clipboard = qapp.clipboard()
    assert clipboard is not None
    saved = clipboard.text()
    try:
        for path in TEXT_ROWS:
            field = child(editor, QLineEdit, f"form.input.{path}")
            text = field.text()
            assert field.isEnabled(), f"{path} must not be greyed out"
            type_text(field, "9")
            key_click(field, Qt.Key.Key_Backspace)
            key_click(field, Qt.Key.Key_Return)
            assert field.text() == text
            key_click(field, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            assert field.selectedText() == text
            key_click(field, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
            assert clipboard.text() == text
    finally:
        clipboard.setText(saved)
    assert state.input.data == DATA


def test_dropdowns_and_checkboxes_ignore_clicks_wheel_and_keys(view: View) -> None:
    editor, state = view
    for path in DROPDOWN_ROWS:
        dropdown = child(editor, QComboBox, f"form.input.{path}")
        shown = dropdown.currentText()
        assert dropdown.isEnabled(), f"{path} must not be greyed out"
        assert not opens_popup(dropdown), path
        wheel(dropdown)
        key_click(dropdown, Qt.Key.Key_Down)
        key_click(dropdown, Qt.Key.Key_Up)
        assert dropdown.currentText() == shown, path
    for box in (
        child(editor, QCheckBox, "form.input.flag"),
        enabled_box(editor, "extra"),
    ):
        assert box.isEnabled(), f"{box.objectName()} must not be greyed out"
        click(box)
        key_click(box, Qt.Key.Key_Space, text=" ")
        assert not box.isChecked(), box.objectName()
    assert state.input.data == DATA


def test_expression_action_is_disabled_and_the_menu_offers_only_copying(
    view: View,
) -> None:
    editor, state = view
    action = mode(editor, "amount")
    assert not action.isEnabled()
    action.trigger()
    assert not action.isChecked()
    assert state.input.data == DATA
    field = child(editor, QLineEdit, "form.input.amount")
    # Copy is offered once there is a selection, as in any text field.
    key_click(field, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    entries = menu_entries(field)
    assert ("Expression", False) in entries
    usable = {text for text, enabled in entries if enabled}
    assert "Copy" in usable
    assert usable <= {"Copy", "Select All"}


def test_folds_still_expand_and_collapse_without_writing(
    view: View, qapp: QApplication
) -> None:
    editor, state = view
    for block, inner in (
        (child(editor, QWidget, "form.group.Advanced"), "form.input.phase"),
        (row_of(editor, "section"), "form.input.section.level"),
    ):
        button = fold(block)
        content = child(editor, QLineEdit, inner)
        assert button.isEnabled() and not content.isVisibleTo(editor)
        button.click()
        qapp.processEvents()
        assert content.isVisibleTo(editor), inner
        assert content.isEnabled(), f"{inner} must not be greyed out"
        button.click()
        qapp.processEvents()
        assert not content.isVisibleTo(editor), inner
    assert state.input.data == DATA


def test_external_updates_show_while_read_only(view: View) -> None:
    editor, state = view
    state.write({("amount",): 6.0, ("mode",): "slow"}, FormEnv())
    assert float(child(editor, QLineEdit, "form.input.amount").text()) == 6.0
    assert child(editor, QComboBox, "form.input.mode").currentText() == "slow"


def test_enabling_again_restores_every_edit_and_mode_switch(
    view: View, qapp: QApplication
) -> None:
    editor, state = view
    editor.set_editing_enabled(enabled=True)
    qapp.processEvents()

    amount = child(editor, QLineEdit, "form.input.amount")
    key_click(amount, Qt.Key.Key_End)
    type_text(amount, "5")
    key_click(amount, Qt.Key.Key_Return)
    assert state.input.data["amount"] == 2.05
    # The write keeps the form's shape, so the typed-into field stays.
    assert child(editor, QLineEdit, "form.input.amount") is amount
    mode(editor, "amount").trigger()
    assert state.input.data["amount"] == {"expr": "2.05"}

    label = child(editor, QLineEdit, "form.input.label")
    key_click(label, Qt.Key.Key_End)
    key_click(label, Qt.Key.Key_Backspace)
    key_click(label, Qt.Key.Key_Return)
    assert state.input.data["label"] == "prob"

    choice = child(editor, QComboBox, "form.input.mode")
    wheel(choice)
    assert state.input.data["mode"] == "slow"
    key_click(choice, Qt.Key.Key_Up)
    assert state.input.data["mode"] == "fast"
    assert opens_popup(choice)

    drive = child(editor, QComboBox, "form.input.drive")
    key_click(drive, Qt.Key.Key_Down)
    assert state.input.data["drive"] == "Q1.pulses.pi2"

    flag = child(editor, QCheckBox, "form.input.flag")
    click(flag)
    assert state.input.data["flag"] is True
    key_click(flag, Qt.Key.Key_Space, text=" ")
    assert state.input.data["flag"] is False

    click(enabled_box(editor, "extra"))
    assert state.input.data["extra"] is not None
