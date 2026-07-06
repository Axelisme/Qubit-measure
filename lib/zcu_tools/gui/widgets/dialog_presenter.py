"""Injectable dialog behavior for GUI widgets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QMessageBox,
    QPushButton,
    QWidget,
)

from zcu_tools.gui.widgets.dialog_lifecycle import DialogRefStore


class DialogPresenter(Protocol):
    """Small dialog port used by widgets that need scriptable test behavior."""

    def information(self, parent: QWidget, title: str, message: str) -> None: ...

    def warning(self, parent: QWidget, title: str, message: str) -> None: ...

    def critical(self, parent: QWidget, title: str, message: str) -> None: ...

    def confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        default: bool = False,
    ) -> bool: ...

    def confirm_async(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        on_decision: Callable[[bool], None],
        default: bool = False,
    ) -> None: ...

    def destructive_confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        action_text: str,
        on_decision: Callable[[bool], None],
        default: bool = False,
    ) -> None: ...


class QtDialogPresenter:
    """Production DialogPresenter backed by QMessageBox."""

    def __init__(self, dialog_refs: DialogRefStore | None = None) -> None:
        self._dialog_refs = dialog_refs
        self._owned_dialog_refs = DialogRefStore()

    def information(self, parent: QWidget, title: str, message: str) -> None:
        if self._dialog_refs is None:
            QMessageBox.information(parent, title, message)
            return
        box = self._message_box(parent, QMessageBox.Icon.Information, title, message)
        self._dialog_refs.open_transient(box)

    def warning(self, parent: QWidget, title: str, message: str) -> None:
        if self._dialog_refs is None:
            QMessageBox.warning(parent, title, message)
            return
        box = self._message_box(parent, QMessageBox.Icon.Warning, title, message)
        self._dialog_refs.open_transient(box)

    def critical(self, parent: QWidget, title: str, message: str) -> None:
        if self._dialog_refs is None:
            QMessageBox.critical(parent, title, message)
            return
        box = self._message_box(parent, QMessageBox.Icon.Critical, title, message)
        self._dialog_refs.open_transient(box)

    def confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        default: bool = False,
    ) -> bool:
        yes = QMessageBox.StandardButton.Yes
        no = QMessageBox.StandardButton.No
        answer = QMessageBox.question(
            parent,
            title,
            message,
            yes | no,
            yes if default else no,
        )
        return answer == yes

    def confirm_async(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        on_decision: Callable[[bool], None],
        default: bool = False,
    ) -> None:
        box = self._message_box(parent, QMessageBox.Icon.Question, title, message)
        yes_button = self._require_button(
            box.addButton(QMessageBox.StandardButton.Yes), "Yes"
        )
        no_button = self._require_button(
            box.addButton(QMessageBox.StandardButton.No), "No"
        )
        box.setDefaultButton(yes_button if default else no_button)
        self._open_decision_box(box, yes_button, on_decision)

    def destructive_confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        action_text: str,
        on_decision: Callable[[bool], None],
        default: bool = False,
    ) -> None:
        box = self._message_box(parent, QMessageBox.Icon.Warning, title, message)
        action_button = self._require_button(
            box.addButton(action_text, QMessageBox.ButtonRole.DestructiveRole),
            action_text,
        )
        cancel_button = self._require_button(
            box.addButton(QMessageBox.StandardButton.Cancel), "Cancel"
        )
        if default:
            box.setDefaultButton(action_button)
        else:
            box.setDefaultButton(cancel_button)
        self._open_decision_box(box, action_button, on_decision)

    def _open_decision_box(
        self,
        box: QMessageBox,
        accept_button: QPushButton,
        on_decision: Callable[[bool], None],
    ) -> None:
        def _on_finished(_status: int) -> None:
            on_decision(box.clickedButton() is accept_button)

        refs = (
            self._dialog_refs
            if self._dialog_refs is not None
            else self._owned_dialog_refs
        )
        refs.open_transient(box, on_finished=_on_finished)

    @staticmethod
    def _require_button(button: QPushButton | None, label: str) -> QPushButton:
        if button is None:
            raise RuntimeError(f"Qt did not create message-box button: {label}")
        return button

    @staticmethod
    def _message_box(
        parent: QWidget,
        icon: QMessageBox.Icon,
        title: str,
        message: str,
    ) -> QMessageBox:
        box = QMessageBox(parent)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(message)
        return box
