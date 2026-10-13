"""Deliver keyboard and mouse input to one widget, as a user's would arrive.

QTest's PyQt6 stubs read its functions as unbound instance methods, so GUI
tests send the same Qt events through QApplication.sendEvent instead.
"""

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QWidget


def key_click(
    widget: QWidget,
    key: Qt.Key,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    text: str = "",
) -> None:
    """Press and release key on widget; text is the character it types, if any."""
    for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
        QApplication.sendEvent(widget, QKeyEvent(kind, key, modifiers, text))


def type_text(widget: QWidget, text: str) -> None:
    """Type text into widget one digit or letter key at a time."""
    for char in text:
        key_click(widget, Qt.Key(ord(char.upper())), text=char)


def mouse_click(widget: QWidget, at: QPoint) -> None:
    """Press and release the left mouse button at widget coordinates at."""
    local = QPointF(at)
    screen = QPointF(widget.mapToGlobal(at))
    left = Qt.MouseButton.LeftButton
    for kind, buttons in (
        (QEvent.Type.MouseButtonPress, left),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(
            widget,
            QMouseEvent(
                kind, local, screen, left, buttons, Qt.KeyboardModifier.NoModifier
            ),
        )


def mouse_double_click(widget: QWidget, at: QPoint) -> None:
    """Double-click the left mouse button at widget coordinates at.

    Qt delivers a double click as press, release, double-click, release; the
    first press and release alone are a single click.
    """
    local = QPointF(at)
    screen = QPointF(widget.mapToGlobal(at))
    left = Qt.MouseButton.LeftButton
    for kind, buttons in (
        (QEvent.Type.MouseButtonPress, left),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
        (QEvent.Type.MouseButtonDblClick, left),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(
            widget,
            QMouseEvent(
                kind, local, screen, left, buttons, Qt.KeyboardModifier.NoModifier
            ),
        )
