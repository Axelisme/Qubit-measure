"""Process-wide mouse and keyboard text selection for QMessageBox content."""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QApplication, QLabel, QMessageBox, QTextEdit

_SELECTABLE = (
    Qt.TextInteractionFlag.TextSelectableByMouse
    | Qt.TextInteractionFlag.TextSelectableByKeyboard
)
_SELECTOR_OBJECT_NAME = "zcu_tools_selectable_message_box_text"


class _MessageBoxTextSelector(QObject):
    """Application event filter that makes a QMessageBox's text selectable on Show."""

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if (
            a1 is not None
            and a1.type() == QEvent.Type.Show
            and isinstance(a0, QMessageBox)
        ):
            _make_message_box_text_selectable(a0)
        return super().eventFilter(a0, a1)


def install_selectable_message_boxes(app: QApplication) -> QObject:
    """Make the text of every QMessageBox in ``app`` selectable when it is shown.

    Installs an event filter on ``app``. Each time any QMessageBox receives a
    Show event, its text labels (main and informative text) and its detailed
    text area gain ``TextSelectableByMouse | TextSelectableByKeyboard``; existing
    interaction flags such as link access are kept. Other widgets are not
    changed, and the box's own buttons and behavior stay as they are. Text a box
    adds after it is already visible becomes selectable at its next Show.

    The call is idempotent: if ``app`` already has the filter, it returns that
    same filter and installs nothing new. The returned filter is a child of
    ``app``; deleting it (for example with ``deleteLater()``) uninstalls the
    behavior. Call it on the GUI thread that owns ``app``.
    """
    # findChildren, not findChild: the PyQt6 stubs type findChild as never None.
    installed = app.findChildren(
        QObject,
        _SELECTOR_OBJECT_NAME,
        Qt.FindChildOption.FindDirectChildrenOnly,
    )
    if installed:
        return installed[0]
    selector = _MessageBoxTextSelector(app)
    selector.setObjectName(_SELECTOR_OBJECT_NAME)
    app.installEventFilter(selector)
    return selector


def _make_message_box_text_selectable(box: QMessageBox) -> None:
    for label in box.findChildren(QLabel):
        # The icon label shows only a pixmap; text labels carry the message.
        if label.text():
            label.setTextInteractionFlags(label.textInteractionFlags() | _SELECTABLE)
    for text_edit in box.findChildren(QTextEdit):
        text_edit.setTextInteractionFlags(
            text_edit.textInteractionFlags() | _SELECTABLE
        )


__all__ = ["install_selectable_message_boxes"]
