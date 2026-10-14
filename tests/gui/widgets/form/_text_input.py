"""Finish an entry in a FormEditor text row, which commits on Enter."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLineEdit

from tests.gui._user_input import key_click


def commit_text(edit: QLineEdit, text: str) -> None:
    """Set edit's whole text and press Enter, as a user finishing an entry."""
    edit.setText(text)
    key_click(edit, Qt.Key.Key_Return)
