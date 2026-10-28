"""Shared caption, uniqueness and Qt availability rules for component input."""

from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDateTimeEdit,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)


def display_text(widget: QWidget) -> str:
    """Read a supported control's caption/value, or an empty string for other kinds."""
    if isinstance(widget, (QAbstractButton, QLabel, QLineEdit, QDateTimeEdit)):
        return widget.text()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, (QTextEdit, QPlainTextEdit)):
        return widget.toPlainText()
    return ""


def unique[T](query: str, candidates: list[T], descriptions: list[str]) -> T:
    """Return the sole candidate or raise LookupError with query and descriptions."""
    if len(candidates) != 1:
        raise LookupError(
            f"{query!r}: {len(candidates)} candidates: {', '.join(descriptions)}"
        )
    return candidates[0]


def on_screen(widget: QWidget) -> bool:
    """Check Qt visibility and a nonempty visible region, not OS-level occlusion."""
    return widget.isVisible() and not widget.visibleRegion().isEmpty()


def can_interact(widget: QWidget) -> bool:
    """Reject hidden/disabled widgets and widgets outside the active modal's ownership."""
    if not on_screen(widget) or not widget.isEnabled():
        return False
    modal = QApplication.activeModalWidget()
    if modal is None:
        return True
    # Qt excludes child windows (including combo popups) from isAncestorOf.
    # Only controls whose ownership chain reaches the active modal may act.
    parent = widget
    while parent is not None:
        if parent is modal:
            return True
        parent = parent.parentWidget()
    return False
