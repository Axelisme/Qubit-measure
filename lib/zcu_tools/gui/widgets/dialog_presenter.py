"""Injectable dialog behavior for GUI widgets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from qtpy.QtWidgets import (
    QMessageBox,
    QPushButton,
    QWidget,
)

from zcu_tools.gui.widgets.dialog_lifecycle import DialogRefStore

# The answer to an unsaved-changes question: apply the changes first, drop
# them, or abandon the action that asked.
type SaveDiscardCancel = Literal["save", "discard", "cancel"]


class DialogPresenter(Protocol):
    """Present messages and user decisions without prescribing a dialog backend."""

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

    def save_discard_cancel(
        self, parent: QWidget, title: str, message: str
    ) -> SaveDiscardCancel:
        """Ask modally whether to save, discard or keep unsaved changes.

        Blocks until the user answers and returns "save", "discard" or
        "cancel"; closing the question answers "cancel", the default.
        """
        ...

    def destructive_confirm(
        self,
        parent: QWidget,
        title: str,
        message: str,
        *,
        action_text: str,
        on_decision: Callable[[bool], None],
        default: bool = False,
        cancel_text: str = "Cancel",
    ) -> None:
        """Present a destructive action and a labelled non-destructive alternative.

        Deliver True only for the explicit action; alternative or dismissal delivers
        False. ``default`` selects the action instead of the alternative as default.
        """
        ...


class QtDialogPresenter:
    """Production DialogPresenter backed by QMessageBox."""

    def __init__(
        self,
        dialog_refs: DialogRefStore | None = None,
        *,
        non_modal_notifications: bool = False,
    ) -> None:
        """Select notification policy without changing modal decision prompts.

        non_modal_notifications retains and shows information/warning/critical
        boxes non-modally, even without a caller-owned ref store. The default
        preserves existing presentation for other applications.
        """
        self._non_modal_notifications = non_modal_notifications
        self._dialog_refs = dialog_refs
        self._owned_dialog_refs = DialogRefStore()

    def information(self, parent: QWidget, title: str, message: str) -> None:
        self._notify(parent, QMessageBox.Icon.Information, title, message)

    def warning(self, parent: QWidget, title: str, message: str) -> None:
        self._notify(parent, QMessageBox.Icon.Warning, title, message)

    def critical(self, parent: QWidget, title: str, message: str) -> None:
        self._notify(parent, QMessageBox.Icon.Critical, title, message)

    def _notify(
        self, parent: QWidget, icon: QMessageBox.Icon, title: str, message: str
    ) -> None:
        box = self._message_box(parent, icon, title, message)
        if self._dialog_refs is None and not self._non_modal_notifications:
            box.exec()
            return
        refs = (
            self._dialog_refs
            if self._dialog_refs is not None
            else self._owned_dialog_refs
        )
        refs.open_transient(box, modal=not self._non_modal_notifications)

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

    def save_discard_cancel(
        self, parent: QWidget, title: str, message: str
    ) -> SaveDiscardCancel:
        save = QMessageBox.StandardButton.Save
        discard = QMessageBox.StandardButton.Discard
        cancel = QMessageBox.StandardButton.Cancel
        box = self._message_box(parent, QMessageBox.Icon.Question, title, message)
        box.addButton(save)
        discard_button = self._require_button(box.addButton(discard), "Discard")
        # Platform themes may label the standard button "Close without Saving".
        discard_button.setText("Discard")
        cancel_button = self._require_button(box.addButton(cancel), "Cancel")
        box.setDefaultButton(cancel_button)
        answer = box.exec()
        if answer == save.value:
            return "save"
        if answer == discard.value:
            return "discard"
        return "cancel"

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
        cancel_text: str = "Cancel",
    ) -> None:
        """Open the labelled action/alternative choice; dismissal delivers False."""
        box = self._message_box(parent, QMessageBox.Icon.Warning, title, message)
        action_button = self._require_button(
            box.addButton(action_text, QMessageBox.ButtonRole.DestructiveRole),
            action_text,
        )
        cancel_button = self._require_button(
            box.addButton(cancel_text, QMessageBox.ButtonRole.RejectRole), cancel_text
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
