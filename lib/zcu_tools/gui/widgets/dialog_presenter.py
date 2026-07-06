"""Injectable dialog behavior for GUI widgets."""

from __future__ import annotations

from typing import Protocol

from qtpy.QtWidgets import QMessageBox, QWidget  # type: ignore[attr-defined]

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

    def destructive_confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        action_text: str,
        default: bool = False,
    ) -> bool: ...


class QtDialogPresenter:
    """Production DialogPresenter backed by QMessageBox."""

    def __init__(self, dialog_refs: DialogRefStore | None = None) -> None:
        self._dialog_refs = dialog_refs

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

    def destructive_confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        action_text: str,
        default: bool = False,
    ) -> bool:
        box = self._message_box(parent, QMessageBox.Icon.Warning, title, message)
        action_button = box.addButton(
            action_text, QMessageBox.ButtonRole.DestructiveRole
        )
        box.addButton(QMessageBox.StandardButton.Cancel)
        if default:
            box.setDefaultButton(action_button)
        else:
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return box.clickedButton() is action_button

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
