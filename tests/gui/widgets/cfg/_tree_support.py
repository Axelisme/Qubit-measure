"""Public Qt observations and input for cfg tree tests."""

from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem
from zcu_tools.gui.widgets.cfg import CfgFormWidget


def tree_widget(form: CfgFormWidget) -> QTreeWidget:
    tree = form.findChild(QTreeWidget)
    assert tree is not None
    return tree


def tree_item(form: CfgFormWidget, path: str) -> QTreeWidgetItem:
    root = tree_widget(form).invisibleRootItem()
    assert root is not None
    pending = [root]
    while pending:
        item = pending.pop()
        if item.data(0, Qt.ItemDataRole.UserRole) == path:
            return item
        for index in range(item.childCount()):
            child = item.child(index)
            if child is not None:
                pending.append(child)
    raise AssertionError(f"no tree item for path {path!r}")


def click_row(
    qapp: QApplication, form: CfgFormWidget, item: QTreeWidgetItem, *, column: int = 0
) -> None:
    form.resize(600, 400)
    form.show()
    qapp.processEvents()
    tree = tree_widget(form)
    tree.scrollToItem(item)
    qapp.processEvents()
    rect = tree.visualItemRect(item)
    assert rect.isValid()
    viewport = tree.viewport()
    assert viewport is not None
    x = tree.columnViewportPosition(column) + tree.columnWidth(column) // 2
    position = QPoint(x, rect.center().y())
    for event_type, buttons in (
        (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        event = QMouseEvent(
            event_type,
            QPointF(position),
            QPointF(viewport.mapToGlobal(position)),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, event)
    qapp.processEvents()
