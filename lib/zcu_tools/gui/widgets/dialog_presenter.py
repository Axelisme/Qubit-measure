"""Injectable dialog behavior for GUI widgets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True, kw_only=True)
class DestructiveConfirmation:
    """Content and labelled alternatives for an explicit destructive decision."""

    title: str
    """Window title of the confirmation prompt."""
    message: str
    """Explanation of the action and its consequences shown in the prompt."""
    action_text: str
    """Label of the only button that accepts the destructive action."""
    default: bool = False
    """True defaults to the action; False defaults to the safe alternative."""
    cancel_text: str = "Cancel"
    """Label for the non-destructive alternative; dismissal also declines."""


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
        confirmation: DestructiveConfirmation,
        *,
        on_decision: Callable[[bool], None],
    ) -> None:
        """Present a destructive action and a labelled non-destructive alternative.

        Deliver True only for the explicit action; alternative or dismissal delivers
        False. ``confirmation.default`` selects the action instead of the
        alternative as default.
        """
        ...


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
        confirmation: DestructiveConfirmation,
        *,
        on_decision: Callable[[bool], None],
    ) -> None:
        """Open the labelled action/alternative choice; dismissal delivers False."""
        box = self._message_box(
            parent, QMessageBox.Icon.Warning, confirmation.title, confirmation.message
        )
        action_button = self._require_button(
            box.addButton(
                confirmation.action_text, QMessageBox.ButtonRole.DestructiveRole
            ),
            confirmation.action_text,
        )
        cancel_button = self._require_button(
            box.addButton(confirmation.cancel_text, QMessageBox.ButtonRole.RejectRole),
            confirmation.cancel_text,
        )
        if confirmation.default:
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
