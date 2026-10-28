"""Deliver real Qt events and visible combo/splitter gestures."""

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QSplitter,
    QStyle,
    QStyleOptionComboBox,
    QWidget,
)

from .selection import can_interact, unique


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
    """Send left click, double-click and release events at widget coordinates."""
    mouse_click(widget, at)
    for kind, buttons in (
        (QEvent.Type.MouseButtonDblClick, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(
            widget,
            QMouseEvent(
                kind,
                QPointF(at),
                QPointF(widget.mapToGlobal(at)),
                Qt.MouseButton.LeftButton,
                buttons,
                Qt.KeyboardModifier.NoModifier,
            ),
        )


def choose_option(widget: QComboBox, value: str) -> None:
    """Choose an exact visible combo option in an already-usable widget.

    Missing/ambiguous value raises LookupError; disabled options are no-ops.
    Use real popup keys and close any popup even on failure."""
    index = unique(
        value,
        [i for i in range(widget.count()) if widget.itemText(i) == value],
        [widget.itemText(i) for i in range(widget.count())],
    )
    model = widget.model()
    if (
        model is None
        or not model.flags(model.index(index, widget.modelColumn()))
        & Qt.ItemFlag.ItemIsEnabled
    ):
        return
    # Editable combos reserve their center for text; only the arrow opens them.
    option = QStyleOptionComboBox()
    option.initFrom(widget)
    option.editable = widget.isEditable()
    style = widget.style()
    if style is None:
        raise RuntimeError("Combo has no style for its arrow geometry")
    arrow = style.subControlRect(
        QStyle.ComplexControl.CC_ComboBox,
        option,
        QStyle.SubControl.SC_ComboBoxArrow,
        widget,
    )
    mouse_click(widget, arrow.center())
    try:
        view = widget.view()
        if view is None:
            raise RuntimeError("Combo has no option view")
        viewport = view.viewport()
        if viewport is None:
            raise RuntimeError("Combo option view has no viewport")
        # The viewport paints the options and covers the view's visible region.
        if not can_interact(viewport):
            return
        key_click(view, Qt.Key.Key_Home)
        for _ in range(index):
            key_click(view, Qt.Key.Key_Down)
        key_click(view, Qt.Key.Key_Return)
    finally:
        popup = QApplication.activePopupWidget()
        if popup is not None:
            key_click(popup, Qt.Key.Key_Escape)
    QApplication.processEvents()


def drag_splitter(splitter: QSplitter, delta: int) -> None:
    """Drag the first splitter handle by delta logical pixels along its axis.

    A missing handle raises ValueError. Unusable handles or clipped endpoints
    are no-ops; the splitter must already have the requested widget type."""
    handle = splitter.handle(1)
    if handle is None:
        raise ValueError("Splitter has no handle")
    if not can_interact(handle):
        return
    start = handle.rect().center()
    end = start + (
        QPoint(delta, 0)
        if splitter.orientation() == Qt.Orientation.Horizontal
        else QPoint(0, delta)
    )
    if not splitter.rect().contains(handle.mapTo(splitter, end)):
        return
    for kind, point, button, buttons in (
        (
            QEvent.Type.MouseButtonPress,
            start,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        ),
        (
            QEvent.Type.MouseMove,
            end,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
        ),
        (
            QEvent.Type.MouseButtonRelease,
            end,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
        ),
    ):
        QApplication.sendEvent(
            handle,
            QMouseEvent(
                kind,
                QPointF(point),
                QPointF(handle.mapToGlobal(point)),
                button,
                buttons,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
    QApplication.processEvents()
