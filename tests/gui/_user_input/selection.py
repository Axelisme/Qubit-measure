"""Shared caption, uniqueness and Qt availability rules for component input."""

from PyQt6 import sip
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDateTimeEdit,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QStatusBar,
    QTextEdit,
    QWidget,
)


def display_text(widget: QWidget) -> str:
    """Read a supported control's caption/value, or an empty string for other kinds."""
    if isinstance(widget, (QAbstractButton, QLabel, QLineEdit, QDateTimeEdit)):
        return widget.text()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QStatusBar):
        return widget.currentMessage()
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


def is_live_widget(widget: QWidget) -> bool:
    """Return whether widget wraps a live Qt QWidget without calling Qt on it.

    Checks SIP wrapper deletion state and Qt's C++ QApplication.allWidgets()
    registry, so mid-walk deletions, dangling wrappers whose C++ deletion SIP
    missed, and cached QWidget aliases for non-widget QObjects all return False
    without dereferencing widget's C++ pointer.
    """
    if sip.isdeleted(widget) or QApplication.instance() is None:
        return False
    return any(live is widget for live in QApplication.allWidgets())


class LiveWidgetSnapshot:
    """Snapshot Qt's live QApplication.allWidgets() set for O(1) walk checks.

    Retains the returned widget list alongside its Python object ids so wrapper
    ids cannot be recycled while this snapshot is active. Call refresh() after
    any step that can trigger Qt style/paint callbacks or event delivery.
    """

    def __init__(self) -> None:
        self._widgets: list[QWidget] = []
        self._ids: set[int] = set()
        self.refresh()

    def refresh(self) -> None:
        """Reload live QWidget instances from Qt's QApplication.allWidgets()."""
        self._widgets = (
            QApplication.allWidgets() if QApplication.instance() is not None else []
        )
        self._ids = {id(widget) for widget in self._widgets}

    def contains(self, widget: QWidget) -> bool:
        """Return whether widget is undeleted and present in this live snapshot."""
        return not sip.isdeleted(widget) and id(widget) in self._ids

    def on_screen(self, widget: QWidget) -> bool:
        """Check live snapshot membership, Qt visibility and nonempty visible region."""
        return (
            self.contains(widget)
            and widget.isVisible()
            and not widget.visibleRegion().isEmpty()
        )

    def descendants[W: QWidget](
        self,
        root: QWidget,
        widget_type: type[W],
        *,
        include_root: bool = False,
        options: Qt.FindChildOption = Qt.FindChildOption.FindChildrenRecursively,
    ) -> list[W]:
        """Return live descendant widgets of root in widget-tree order.

        Filters findChildren results against this live snapshot before callers
        inspect widget properties, excluding cached non-widget aliases and
        deleted wrappers.
        """
        if not self.contains(root):
            return []
        candidates: list[W] = []
        if include_root and isinstance(root, widget_type):
            candidates.append(root)
        candidates.extend(root.findChildren(widget_type, options=options))
        return [widget for widget in candidates if self.contains(widget)]


def on_screen(widget: QWidget) -> bool:
    """Check Qt visibility and a nonempty visible region, not OS-level occlusion.

    Returns False for deleted, dangling or non-widget wrappers before calling
    any Qt method on widget.
    """
    return (
        is_live_widget(widget)
        and widget.isVisible()
        and not widget.visibleRegion().isEmpty()
    )


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
