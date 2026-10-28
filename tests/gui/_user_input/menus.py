"""Deliver real input inside blocking Qt menus and external file choosers."""

import logging
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QContextMenuEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMenu,
    QWidget,
)

from . import observing
from .input import key_click, mouse_click
from .selection import unique


def dismiss_menus() -> None:
    """Press Escape until Qt has no active popup menu."""
    while isinstance(menu := QApplication.activePopupWidget(), QMenu):
        key_click(menu, Qt.Key.Key_Escape)


def select_menu_path(path: tuple[str, ...]) -> None:
    """Click exact visible action captions starting at the active popup menu.

    Missing/ambiguous actions or non-submenu intermediate entries raise
    LookupError. Disabled actions dismiss menus without invoking the action.
    """
    for index, caption in enumerate(path):
        menu = QApplication.activePopupWidget()
        if not isinstance(menu, QMenu):
            raise LookupError(f"No visible popup menu for {caption!r}")
        actions = [a for a in menu.actions() if a.isVisible()]
        action = unique(
            caption,
            [a for a in actions if a.text() == caption],
            [a.text() for a in actions],
        )
        if not action.isEnabled():
            dismiss_menus()
            return
        if index < len(path) - 1 and action.menu() is None:
            raise LookupError(f"{caption!r} is not a submenu")
        mouse_click(menu, menu.actionGeometry(action).center())
        if index < len(path) - 1:
            key_click(menu, Qt.Key.Key_Right)
            QApplication.processEvents()
        # Let the final click unwind QMenu.exec before deferred deletions.
        # Commands may retire their action and control synchronously.


def visit_menu(open_menu: Callable[[], None], visit: Callable[[], None]) -> None:
    """Deliver visit inside the opener's Qt loop; propagate failures after unwind."""
    errors: list[Exception] = []

    def choose() -> None:
        try:
            visit()
        except (LookupError, RuntimeError, ValueError, TypeError, OSError) as error:
            # Return Qt callback failures to the scenario after exec unwinds.
            errors.append(error)
            dismiss_menus()

    def timed_out() -> None:
        errors.append(TimeoutError("Blocking menu input delivery did not finish"))
        dismiss_menus()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(timed_out)
    watchdog.start(30_000)
    QTimer.singleShot(0, choose)
    try:
        open_menu()
    finally:
        watchdog.stop()
    if errors:
        raise errors[0]


def invoke_context_action(
    target: QWidget, point: QPoint, action: str, evidence: Path | None
) -> None:
    """Deliver a context-menu choice at a guarded target; unwind before raising."""
    errors: list[Exception] = []

    def choose() -> None:
        try:
            if evidence is not None:
                menu = QApplication.activePopupWidget()
                if not isinstance(menu, QMenu):
                    raise LookupError("No visible context menu")
                observing.screenshot(menu, evidence)
            select_menu_path((action,))
        except (LookupError, RuntimeError, ValueError, TypeError) as error:
            errors.append(error)
            dismiss_menus()

    def timed_out() -> None:
        errors.append(TimeoutError("Context menu delivery did not finish"))
        dismiss_menus()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(timed_out)
    watchdog.start(30_000)
    QTimer.singleShot(0, choose)
    try:
        QApplication.sendEvent(
            target,
            QContextMenuEvent(
                QContextMenuEvent.Reason.Mouse, point, target.mapToGlobal(point)
            ),
        )
        QApplication.processEvents()
    finally:
        watchdog.stop()
        dismiss_menus()
    if errors:
        raise errors[0]


def deliver_file_choice(
    open_chooser: Callable[[], None], fill_chooser: Callable[[], None]
) -> None:
    """Fill a synchronous opener's file chooser and dismiss it on delivery failure."""
    errors: list[Exception] = []

    def choose() -> None:
        try:
            fill_chooser()
        except Exception as error:
            logging.exception("File chooser input delivery failed")
            errors.append(error)
            dismiss()

    def dismiss() -> None:
        if isinstance(dialog := QApplication.activeModalWidget(), QFileDialog):
            key_click(dialog, Qt.Key.Key_Escape)

    def timed_out() -> None:
        errors.append(TimeoutError("File chooser input delivery did not finish"))
        dismiss()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(timed_out)
    delivery = QTimer()
    delivery.setSingleShot(True)
    delivery.timeout.connect(choose)
    # The synchronous opener enters a nested Qt loop that must deliver choose.
    watchdog.start(30_000)
    delivery.start(0)
    try:
        open_chooser()
    finally:
        delivery.stop()
        watchdog.stop()
    if errors:
        raise errors[0]
