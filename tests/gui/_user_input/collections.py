"""Observe and drive visible table/tree viewports without model mutation."""

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
)

from .._observations import TableRowObservation
from .input import key_click, mouse_click, type_text
from .selection import can_interact, on_screen, unique


def tree_columns(tree: QTreeWidget, item: QTreeWidgetItem) -> tuple[str, ...]:
    """Read item captions by column, using empty strings for hidden/clipped cells."""
    viewport = tree.viewport()
    assert viewport is not None
    row = tree.visualItemRect(item)
    return tuple(
        item.text(i)
        if not tree.isColumnHidden(i)
        and viewport.visibleRegion().intersects(
            QRect(
                tree.columnViewportPosition(i),
                row.y(),
                tree.columnWidth(i),
                row.height(),
            )
        )
        else ""
        for i in range(tree.columnCount())
    )


def visible_tree_items(
    tree: QTreeWidget,
) -> list[tuple[QTreeWidgetItem, tuple[str, ...]]]:
    """Read viewport-intersecting rows and exact visible column-zero caption paths.

    Skip hidden rows and descendants of collapsed rows; never expand or scroll.
    """
    rows: list[tuple[QTreeWidgetItem, tuple[str, ...]]] = []
    viewport = tree.viewport()
    assert viewport is not None

    def visit(item: QTreeWidgetItem, parent: tuple[str, ...]) -> None:
        if item.isHidden():
            return
        path = (*parent, tree_columns(tree, item)[0])
        if viewport.visibleRegion().intersects(tree.visualItemRect(item)):
            rows.append((item, path))
        if item.isExpanded():
            for index in range(item.childCount()):
                child = item.child(index)
                if child is not None:
                    visit(child, path)

    for index in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(index)
        if item is not None:
            visit(item, ())
    return rows


def table_rows(table: QTableWidget) -> tuple[TableRowObservation, ...]:
    """Read rows intersecting the already-visible table viewport."""
    viewport = table.viewport()
    assert viewport is not None
    rows = []
    for row in range(table.rowCount()):
        items = [table.item(row, col) for col in range(table.columnCount())]
        visible = [
            item is not None
            and viewport.visibleRegion().intersects(table.visualItemRect(item))
            for item in items
        ]
        if any(visible):
            rows.append(
                TableRowObservation(
                    columns=tuple(
                        item.text() if item is not None and shown else ""
                        for item, shown in zip(items, visible, strict=True)
                    ),
                    tooltips=tuple(
                        item.toolTip() if item is not None and shown else ""
                        for item, shown in zip(items, visible, strict=True)
                    ),
                    selected=any(
                        item is not None and item.isSelected() for item in items
                    ),
                )
            )
    return tuple(rows)


def select_table_row(
    table: QTableWidget, text: str, *, double_click: bool = False
) -> None:
    """Click a unique fully visible cell in an already-usable table.

    Missing/ambiguous text raises LookupError; double_click adds a double-click event."""
    viewport = table.viewport()
    assert viewport is not None
    items: list[QTableWidgetItem] = []
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            item = table.item(row, column)
            if (
                item is not None
                and item.text() == text
                and viewport.visibleRegion().contains(table.visualItemRect(item))
            ):
                items.append(item)
    item = unique(text, items, [item.text() for item in items])
    point = table.visualItemRect(item).center()
    mouse_click(viewport, point)
    if double_click:
        for kind, buttons in (
            (QEvent.Type.MouseButtonDblClick, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
        ):
            QApplication.sendEvent(
                viewport,
                QMouseEvent(
                    kind,
                    QPointF(point),
                    QPointF(viewport.mapToGlobal(point)),
                    Qt.MouseButton.LeftButton,
                    buttons,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
    QApplication.processEvents()


def drag_column(table: QTableWidget | QTreeWidget, column: int, delta: int) -> None:
    """Drag a table/tree header boundary by delta logical pixels.

    Out-of-range column raises IndexError. Unusable or clipped handles are no-ops."""
    if not 0 <= column < table.columnCount():
        raise IndexError(column)
    header = (
        table.header() if isinstance(table, QTreeWidget) else table.horizontalHeader()
    )
    if header is None:
        raise RuntimeError("Table has no header")
    viewport = header.viewport()
    assert viewport is not None
    start = QPoint(
        header.sectionViewportPosition(column) + header.sectionSize(column) - 1,
        viewport.height() // 2,
    )
    end = start + QPoint(delta, 0)
    if not can_interact(viewport) or not all(
        viewport.visibleRegion().contains(point) for point in (start, end)
    ):
        return
    left = Qt.MouseButton.LeftButton
    for kind, point, button, buttons in (
        (
            QEvent.Type.MouseMove,
            start,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
        ),
        (QEvent.Type.MouseButtonPress, start, left, left),
        (QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, left),
        (QEvent.Type.MouseButtonRelease, end, left, Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(
            viewport,
            QMouseEvent(
                kind,
                QPointF(point),
                QPointF(viewport.mapToGlobal(point)),
                button,
                buttons,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
    QApplication.processEvents()


def expand_tree(tree: QTreeWidget, path: tuple[str, ...]) -> None:
    """Expand an exact visible caption path in an already-usable tree.

    Missing/ambiguous paths raise LookupError; clipped click targets are no-ops."""
    candidates = [
        item for item, captions in visible_tree_items(tree) if captions == path
    ]
    item = unique(
        repr(path),
        candidates,
        [repr(tree_columns(tree, i)) for i in candidates],
    )
    if not item.isExpanded() and item.flags() & Qt.ItemFlag.ItemIsEnabled:
        rect = tree.visualItemRect(item)
        viewport = tree.viewport()
        assert viewport is not None
        target = QPoint(rect.left() + 10, rect.center().y())
        if tree.isColumnHidden(0) or not viewport.visibleRegion().contains(target):
            return
        mouse_click(viewport, target)
        key_click(tree, Qt.Key.Key_Right)
    QApplication.processEvents()


def select_tree_row(tree: QTreeWidget, path: tuple[str, ...]) -> None:
    """Click column zero of an exact visible path in an already-usable tree.

    Missing/ambiguous paths raise LookupError; clipped targets are no-ops."""
    item = unique(
        repr(path),
        [i for i, captions in visible_tree_items(tree) if captions == path],
        [repr(captions) for _, captions in visible_tree_items(tree)],
    )
    viewport = tree.viewport()
    assert viewport is not None
    target = QPoint(
        tree.columnViewportPosition(0) + tree.columnWidth(0) // 2,
        tree.visualItemRect(item).center().y(),
    )
    if viewport.visibleRegion().contains(target):
        mouse_click(viewport, target)
    QApplication.processEvents()


def edit_tree_value(
    tree: QTreeWidget, path: tuple[str, ...], *, value: str, commit: bool = True
) -> None:
    """Edit column one of an exact visible path in an already-usable tree.

    Missing/ambiguous paths or editors raise LookupError. Noneditable or clipped
    cells are no-ops. commit=False leaves the text editor open without Enter."""
    item = unique(
        repr(path),
        [i for i, captions in visible_tree_items(tree) if captions == path],
        [repr(captions) for _, captions in visible_tree_items(tree)],
    )
    if not item.flags() & Qt.ItemFlag.ItemIsEditable:
        return
    rect = tree.visualItemRect(item)
    viewport = tree.viewport()
    assert viewport is not None
    target = QPoint(
        tree.columnViewportPosition(1) + tree.columnWidth(1) // 2,
        rect.center().y(),
    )
    if tree.isColumnHidden(1) or not viewport.visibleRegion().contains(target):
        return
    mouse_click(viewport, target)
    for kind, buttons in (
        (QEvent.Type.MouseButtonDblClick, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(
            viewport,
            QMouseEvent(
                kind,
                QPointF(target),
                QPointF(viewport.mapToGlobal(target)),
                Qt.MouseButton.LeftButton,
                buttons,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
    QApplication.processEvents()
    editors = [e for e in tree.findChildren(QLineEdit) if on_screen(e)]
    editor = unique("tree Value editor", editors, [e.objectName() for e in editors])
    if can_interact(editor):
        key_click(editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        type_text(editor, value)
        if commit:
            key_click(editor, Qt.Key.Key_Return)
    QApplication.processEvents()
