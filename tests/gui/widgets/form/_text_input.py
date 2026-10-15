"""Finish an entry in a FormEditor text row, which commits on Enter."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QLineEdit

from tests.gui._user_input import key_click


def commit_text(edit: QLineEdit, text: str) -> None:
    """Set edit's whole text and press Enter, as a user finishing an entry."""
    edit.setText(text)
    key_click(edit, Qt.Key.Key_Return)


def commit_text_on_focus_loss(edit: QLineEdit, text: str) -> None:
    """Finish a visible dialog field without activating its default button."""
    edit.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()
    edit.setText(text)
    edit.clearFocus()
