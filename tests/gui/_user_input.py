"""Deliver keyboard and mouse input to one widget, as a user's would arrive.

QTest's PyQt6 stubs read its functions as unbound instance methods, so GUI
tests send the same Qt events through QApplication.sendEvent instead.
"""

import logging
from pathlib import Path

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRect, QSize, Qt, QTimer
from PyQt6.QtGui import QContextMenuEvent, QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QScrollArea,
    QStatusBar,
    QStyle,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from ._observations import DialogObservation as DialogObservation
from ._observations import LegendObservation as LegendObservation
from ._observations import ListRowObservation as ListRowObservation
from ._observations import PlotAxesObservation as PlotAxesObservation
from ._observations import PlotLabelBoundsObservation as PlotLabelBoundsObservation
from ._observations import PlotObservation as PlotObservation
from ._observations import ScrollObservation as ScrollObservation
from ._observations import StatusObservation as StatusObservation
from ._observations import TableRowObservation as TableRowObservation
from ._observations import TabObservation as TabObservation
from ._observations import TextInputObservation as TextInputObservation
from ._observations import TreeRowObservation as TreeRowObservation
from ._observations import VisibleObservation as VisibleObservation
from ._observations import WidgetObservation as WidgetObservation


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


def _display_text(widget: QWidget) -> str:
    if isinstance(widget, (QAbstractButton, QLabel, QLineEdit)):
        return widget.text()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, (QTextEdit, QPlainTextEdit)):
        return widget.toPlainText()
    return ""


def _unique[T](query: str, candidates: list[T], descriptions: list[str]) -> T:
    if len(candidates) != 1:
        raise LookupError(
            f"{query!r}: {len(candidates)} candidates: {', '.join(descriptions)}"
        )
    return candidates[0]


class ComponentDriver:
    """Drive a shown window on its Qt owner thread without exposing its objects.

    Text selectors match exact visible captions (or form labels for input).
    ``object_name`` matches exact objectName, preferring on-screen matches.
    With no on-screen match it includes hidden widgets so availability can be
    observed. Multiple on-screen matches or multiple hidden-only matches remain
    ambiguous. Zero or multiple matches raise LookupError
    with candidate descriptions; supply exactly one selector. Operations on
    disabled, hidden, viewport-clipped, or modal-blocked widgets are no-ops.
    On-screen means a nonempty Qt visible region, not OS-level occlusion.
    Clicks and edits use
    mouse/key events, never button.click(), setText(), or application commands.
    The caller owns window/worker startup, completion waits and teardown.
    """

    def __init__(self, window: QWidget) -> None:
        self._window = window

    def _widgets(self) -> list[QWidget]:
        return [self._window, *self._window.findChildren(QWidget)]

    def _find(
        self, text: str | None, object_name: str | None, *, editable: bool = False
    ) -> QWidget:
        if (text is None) == (object_name is None):
            raise ValueError("Supply exactly one of text or object_name")
        candidates = []
        for widget in self._widgets():
            if editable and not isinstance(widget, QLineEdit):
                continue
            if object_name is not None:
                matches = widget.objectName() == object_name
            else:
                caption = (
                    self._field_label(widget) if editable else _display_text(widget)
                )
                matches = self._on_screen(widget) and caption == text
            if matches:
                candidates.append(widget)
        if object_name is not None:
            on_screen = [widget for widget in candidates if self._on_screen(widget)]
            if on_screen:
                candidates = on_screen
        descriptions = [
            f"{type(w).__name__}(objectName={w.objectName()!r}, text={_display_text(w)!r})"
            for w in candidates
        ]
        return _unique(text or object_name or "", candidates, descriptions)

    def _field_label(self, field: QWidget) -> str:
        for widget in self._widgets():
            if isinstance(widget, QLabel) and widget.buddy() is field:
                return widget.text()
            layout = widget.layout()
            if isinstance(layout, QFormLayout):
                label = layout.labelForField(field)
                if isinstance(label, QLabel):
                    return label.text()
        return ""

    @staticmethod
    def _on_screen(widget: QWidget) -> bool:
        return widget.isVisible() and not widget.visibleRegion().isEmpty()

    @staticmethod
    def _can_interact(widget: QWidget) -> bool:
        if not ComponentDriver._on_screen(widget) or not widget.isEnabled():
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

    def resize_window(self, width: int, height: int) -> None:
        """Resize the shown window in logical pixels and flush its Qt layout.

        Hidden or modal-blocked windows are not resized. Nonpositive dimensions
        raise ValueError. Qt may enforce the window's minimum size.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Window dimensions must be positive")
        if self._can_interact(self._window):
            self._window.resize(width, height)
        QApplication.processEvents()

    def window_size(self) -> QSize:
        """Read the shown window's client width/height in device-independent pixels.

        Return a detached QSize value, excluding the native window frame.
        A hidden window raises LookupError rather than supplying unseen geometry.
        This is an observation only; it never resizes the window.
        """
        if not self._window.isVisible():
            raise LookupError("window is not visible")
        return self._window.size()

    def observe(
        self, text: str | None = None, *, object_name: str | None = None
    ) -> WidgetObservation:
        """Read one widget's display/availability/help; lookup errors fail fast."""
        widget = self._find(text, object_name)
        return self._observe_widget(widget)

    @staticmethod
    def _observe_widget(widget: QWidget) -> WidgetObservation:
        on_screen = ComponentDriver._on_screen(widget)
        return WidgetObservation(
            kind=type(widget).__name__,
            text=_display_text(widget) if on_screen else "",
            enabled=widget.isEnabled(),
            visible=on_screen,
            fully_visible=on_screen and widget.visibleRegion().contains(widget.rect()),
            object_name=widget.objectName(),
            tooltip=widget.toolTip() if on_screen else "",
            bounds=(
                widget.mapToGlobal(QPoint()).x(),
                widget.mapToGlobal(QPoint()).y(),
                widget.width(),
                widget.height(),
            )
            if on_screen
            else None,
            caption_fits=(
                on_screen
                and (
                    not isinstance(widget, QAbstractButton)
                    or widget.fontMetrics().horizontalAdvance(widget.text())
                    <= widget.contentsRect().width()
                )
            ),
            checked=(
                widget.isChecked()
                if on_screen
                and isinstance(widget, QAbstractButton)
                and widget.isCheckable()
                else None
            ),
        )

    def list_rows(self) -> tuple[ListRowObservation, ...]:
        """Observe rows intersecting visible QListWidget viewports in this window."""
        rows: list[ListRowObservation] = []
        for listing in self._window.findChildren(QListWidget):
            if not self._on_screen(listing):
                continue
            viewport = listing.viewport()
            assert viewport is not None
            for index in range(listing.count()):
                item = listing.item(index)
                if item is not None and listing.visualItemRect(item).intersects(
                    viewport.rect()
                ):
                    rows.append(
                        ListRowObservation(
                            item.text(), item.toolTip(), item.isSelected()
                        )
                    )
        return tuple(rows)

    def select_list_row(self, text: str) -> None:
        """Click the unique visible row with this exact rendered caption."""
        matches: list[tuple[QListWidget, QListWidgetItem]] = []
        for listing in self._window.findChildren(QListWidget):
            if not self._can_interact(listing):
                continue
            viewport = listing.viewport()
            assert viewport is not None
            for index in range(listing.count()):
                item = listing.item(index)
                if (
                    item is not None
                    and item.text() == text
                    and viewport.rect().contains(listing.visualItemRect(item))
                ):
                    matches.append((listing, item))
        listing, item = _unique(text, matches, [i.text() for _, i in matches])
        viewport = listing.viewport()
        assert viewport is not None
        mouse_click(viewport, listing.visualItemRect(item).center())

    def table_rows(self, *, object_name: str) -> tuple[TableRowObservation, ...]:
        """Observe rows intersecting the named QTableWidget viewport."""
        table = self._find(None, object_name)
        if not isinstance(table, QTableWidget):
            raise TypeError(f"{object_name!r} is not a QTableWidget")
        if not self._on_screen(table):
            return ()
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

    def select_table_row(self, text: str, *, object_name: str) -> None:
        """Click the unique fully visible table cell with this exact caption.

        Disabled, hidden, clipped and modal-blocked tables cannot be clicked.
        The target must already be in the viewport; this does not auto-scroll.
        """
        table = self._find(None, object_name)
        if not isinstance(table, QTableWidget):
            raise TypeError(f"{object_name!r} is not a QTableWidget")
        if not self._can_interact(table):
            return
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
        item = _unique(text, items, [item.text() for item in items])
        mouse_click(viewport, table.visualItemRect(item).center())
        QApplication.processEvents()

    def visible(self) -> VisibleObservation:
        """Snapshot all visible widgets for completion predicates with changing captions."""
        return VisibleObservation(
            widgets=tuple(
                self._observe_widget(w) for w in self._widgets() if self._on_screen(w)
            ),
            tree_rows=tuple(
                TreeRowObservation(
                    tree.objectName(),
                    path,
                    self._tree_columns(tree, item),
                    item.isExpanded(),
                )
                for tree in self._window.findChildren(QTreeWidget)
                if self._on_screen(tree)
                for item, path in self._visible_tree_items(tree)
            ),
        )

    @staticmethod
    def _tree_columns(tree: QTreeWidget, item: QTreeWidgetItem) -> tuple[str, ...]:
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

    @staticmethod
    def _visible_tree_items(
        tree: QTreeWidget,
    ) -> list[tuple[QTreeWidgetItem, tuple[str, ...]]]:
        rows: list[tuple[QTreeWidgetItem, tuple[str, ...]]] = []
        viewport = tree.viewport()
        assert viewport is not None

        def visit(item: QTreeWidgetItem, parent: tuple[str, ...]) -> None:
            if item.isHidden():
                return
            path = (*parent, ComponentDriver._tree_columns(tree, item)[0])
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

    def expand_tree(self, path: tuple[str, ...], *, object_name: str) -> None:
        """Expand an on-screen QTreeWidget row by clicking it then pressing Right.

        path is the exact root-to-row column-zero caption tuple, not model keys.
        Unseen captions are empty strings, never wildcards. A matched row whose
        column-zero click target is outside the visible viewport is a no-op.
        Only rows intersecting the viewport are candidates: collapsed ancestors,
        hidden rows and off-screen rows are not bypassed or scrolled into view.
        Missing/ambiguous trees or visible paths raise LookupError with candidates;
        a non-tree selector raises TypeError. Disabled, hidden or modal-blocked
        trees are no-ops. Already-expanded rows are left unchanged.
        """
        tree = self._find(None, object_name)
        if not isinstance(tree, QTreeWidget):
            raise TypeError(f"{object_name!r} is not a QTreeWidget")
        if not self._can_interact(tree):
            return
        candidates = [
            item
            for item, captions in self._visible_tree_items(tree)
            if captions == path
        ]
        item = _unique(
            repr(path),
            candidates,
            [repr(self._tree_columns(tree, i)) for i in candidates],
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

    def edit_tree_value(
        self, path: tuple[str, ...], *, object_name: str, value: str
    ) -> None:
        """Edit a visible tree row's Value cell using double-click and Enter.

        path contains exact visible root-to-row captions, not model keys.
        object_name selects one QTreeWidget whose second column is Value.
        value is the text to type into its QLineEdit delegate. Missing or
        ambiguous trees/visible paths/editors raise LookupError; a non-tree
        target raises TypeError. The delegate must expose a QLineEdit.
        Disabled, hidden, clipped,
        modal-blocked or noneditable cells are no-ops. Ancestors must already
        be expanded; this method never scrolls or opens hidden rows.
        """
        tree = self._find(None, object_name)
        if not isinstance(tree, QTreeWidget):
            raise TypeError(f"{object_name!r} is not a QTreeWidget")
        if not self._can_interact(tree):
            return
        item = _unique(
            repr(path),
            [i for i, captions in self._visible_tree_items(tree) if captions == path],
            [repr(captions) for _, captions in self._visible_tree_items(tree)],
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
        editors = [e for e in tree.findChildren(QLineEdit) if self._on_screen(e)]
        editor = _unique(
            "tree Value editor", editors, [e.objectName() for e in editors]
        )
        if self._can_interact(editor):
            key_click(editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            type_text(editor, value)
            key_click(editor, Qt.Key.Key_Return)
        QApplication.processEvents()

    def click(self, text: str | None = None, *, object_name: str | None = None) -> None:
        """Click a usable widget with real mouse events.

        Checkboxes use the style's indicator centre: a stretched checkbox's
        widget centre can be outside Qt's clickable label/indicator region.
        Other widgets use their rectangle centre. Availability rules are unchanged.
        """
        widget = self._find(text, object_name)
        if self._can_interact(widget):
            point = widget.rect().center()
            if isinstance(widget, QCheckBox):
                option = QStyleOptionButton()
                option.initFrom(widget)
                option.text = widget.text()
                style = widget.style()
                assert style is not None
                point = style.subElementRect(
                    QStyle.SubElement.SE_CheckBoxIndicator, option, widget
                ).center()
            mouse_click(widget, point)
        QApplication.processEvents()

    def expand_section(self, object_name: str) -> None:
        """Click a visible folded section title; leave an expanded title open.

        The caller must reveal the title first (including its ancestors and tab).
        Missing, clipped, disabled or non-fold titles fail rather than bypass UI.
        """
        title = self.observe(object_name=object_name)
        assert title.visible and title.enabled, object_name
        assert title.text.startswith(("\u25b8 ", "\u25be ")), title
        if title.text.startswith("\u25b8 "):
            self.click(object_name=object_name)
        assert self.observe(object_name=object_name).text.startswith("\u25be ")

    def click_tooltip(self, tooltip: str) -> None:
        """Click the unique on-screen control with this exact tooltip.

        Missing or ambiguous matches raise LookupError. Disabled, clipped or
        modal-blocked controls are no-ops, just as with caption-based click.
        """
        candidates = [
            widget
            for widget in self._widgets()
            if self._on_screen(widget) and widget.toolTip() == tooltip
        ]
        widget = _unique(tooltip, candidates, [w.objectName() for w in candidates])
        if self._can_interact(widget):
            mouse_click(widget, widget.rect().center())
        QApplication.processEvents()

    def _field_control(self, text: str | None, object_name: str | None) -> QWidget:
        if (text is None) == (object_name is None):
            raise ValueError("Supply exactly one of text or object_name")
        if text is not None:
            fields = [
                widget
                for widget in self._widgets()
                if self._on_screen(widget) and self._field_label(widget) == text
            ]
            if fields:
                return _unique(text, fields, [widget.objectName() for widget in fields])
        return self._find(text, object_name)

    def choose_option(
        self,
        text: str | None = None,
        value: str = "",
        *,
        object_name: str | None = None,
    ) -> None:
        """Choose an exact visible option caption using real combo keyboard events.

        Supply exactly one exact text/form-label or object_name selector. Missing
        or ambiguous widget/option captions raise LookupError; a non-combo target
        raises TypeError. Disabled, hidden, clipped or modal-blocked targets and
        disabled options are no-ops. value is a caption, never userData or index.
        No target is shown or scrolled into view. If activation does not open an
        interactable popup, selection stays unchanged. Popups close on failure.
        """
        widget = self._field_control(text, object_name)
        if not isinstance(widget, QComboBox):
            raise TypeError("choose_option requires QComboBox")
        if not self._can_interact(widget):
            return
        index = _unique(
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
            if not self._can_interact(viewport):
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

    def invoke_context_action(
        self,
        text: str | None = None,
        action: str = "",
        *,
        object_name: str | None = None,
        evidence: Path | None = None,
    ) -> None:
        """Open a control's context menu and click an exact visible action caption.

        Supply exactly one exact text/form-label or object_name selector. Missing/
        ambiguous captions raise LookupError. Unavailable widgets and disabled actions are no-ops. Hidden
        actions are not selected. Real context-menu/mouse events are delivered;
        actions and product handlers are never called directly. A scheduled menu
        delivery has a 30-second watchdog; all popup/callback failures are returned
        to the caller after closing the menu, never left in Qt's event loop.
        evidence optionally captures the visible popup before selecting the action.
        """
        widget = self._field_control(text, object_name)
        if not self._can_interact(widget):
            return
        errors: list[Exception] = []

        def choose() -> None:
            try:
                if evidence is not None:
                    menu = QApplication.activePopupWidget()
                    if not isinstance(menu, QMenu):
                        raise LookupError("No visible context menu")
                    ComponentDriver(menu).screenshot(evidence)
                self._select_menu_path((action,))
            except (LookupError, RuntimeError, ValueError, TypeError) as error:
                errors.append(error)
                self._dismiss_menus()

        def timed_out() -> None:
            errors.append(TimeoutError("Context menu delivery did not finish"))
            self._dismiss_menus()

        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(timed_out)
        watchdog.start(30_000)
        QTimer.singleShot(0, choose)
        try:
            point = widget.rect().center()
            QApplication.sendEvent(
                widget,
                QContextMenuEvent(
                    QContextMenuEvent.Reason.Mouse, point, widget.mapToGlobal(point)
                ),
            )
            QApplication.processEvents()
        finally:
            watchdog.stop()
            self._dismiss_menus()
        if errors:
            raise errors[0]

    def select_menu(self, opener: str, path: tuple[str, ...]) -> None:
        """Click opener and select exact captions in its blocking QMenu.

        A queued owner event delivers real mouse/key input inside QMenu.exec().
        Each caption must uniquely identify a visible action in that menu;
        missing/ambiguous captions raise LookupError after dismissing the menu.
        Disabled openers/actions are no-ops. Intermediate captions must be
        submenus. The 30-second watchdog guards the already-scheduled delivery,
        not menu performance. No QAction.trigger or target GUI patching is used.
        """
        if not path:
            raise ValueError("Supply a nonempty menu caption path")
        button = self._find(opener, None)
        if not self._can_interact(button):
            return
        errors: list[Exception] = []

        def choose() -> None:
            try:
                self._select_menu_path(path)
            except (LookupError, RuntimeError, ValueError) as error:
                # Return Qt callback failures to the scenario after exec unwinds.
                errors.append(error)
                self._dismiss_menus()

        def timed_out() -> None:
            errors.append(TimeoutError("Blocking menu input delivery did not finish"))
            self._dismiss_menus()

        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(timed_out)
        watchdog.start(30_000)
        QTimer.singleShot(0, choose)
        try:
            self.click(opener)
        finally:
            watchdog.stop()
        if errors:
            raise errors[0]

    @staticmethod
    def _dismiss_menus() -> None:
        while isinstance(menu := QApplication.activePopupWidget(), QMenu):
            key_click(menu, Qt.Key.Key_Escape)

    @staticmethod
    def _select_menu_path(path: tuple[str, ...]) -> None:
        for index, caption in enumerate(path):
            menu = QApplication.activePopupWidget()
            if not isinstance(menu, QMenu):
                raise LookupError(f"No visible popup menu for {caption!r}")
            actions = [a for a in menu.actions() if a.isVisible()]
            action = _unique(
                caption,
                [a for a in actions if a.text() == caption],
                [a.text() for a in actions],
            )
            if not action.isEnabled():
                ComponentDriver._dismiss_menus()
                return
            if index < len(path) - 1 and action.menu() is None:
                raise LookupError(f"{caption!r} is not a submenu")
            mouse_click(menu, menu.actionGeometry(action).center())
            if index < len(path) - 1:
                key_click(menu, Qt.Key.Key_Right)
                QApplication.processEvents()
            # Let the final click unwind QMenu.exec before deferred deletions.
            # Commands may retire their action and control synchronously.

    def choose_file(
        self, opener: str, path: Path, *, evidence: Path | None = None
    ) -> None:
        """Click opener, enter path in its modal Qt file chooser, and click Open.

        The opener must synchronously open a non-native QFileDialog with Qt's
        visible fileNameEdit. Input paths containing hidden parent directories
        are not supported. The caller supplies an existing file path (spaces
        allowed). Relative paths require cwd and the chooser's initial directory
        to refer to the same directory, configured by the caller before opening.
        Before looking up the opener or applying unavailable no-ops,
        nonexistent paths raise FileNotFoundError and existing non-files raise
        ValueError. Visible caption matches take precedence over hidden copies;
        if none are on screen, a unique unseen opener is a no-op (multiple or
        missing matches still raise LookupError).
        If evidence is a PNG path, capture the visible file chooser
        there before typing or accepting; None skips capture. Capture creates
        parent directories and raises OSError on save failure. Missing/ambiguous
        widgets or a different modal raise LookupError. Disabled/hidden/clipped/
        modal-blocked openers are no-ops, including no evidence capture.
        A queued event supplies real input inside the modal loop; a 30-second
        watchdog guards that delivery. On delivery failure Escape dismisses the
        chooser before the exception returns. No dialog return value is patched.
        """
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        matches = [w for w in self._widgets() if _display_text(w) == opener]
        visible = [w for w in matches if self._on_screen(w)]
        candidates = visible or matches
        button = _unique(
            opener,
            candidates,
            [f"{type(w).__name__}(objectName={w.objectName()!r})" for w in candidates],
        )
        if not self._can_interact(button):
            return
        errors: list[Exception] = []

        def choose() -> None:
            try:
                self._choose_visible_file(path, evidence)
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
            self.click(opener)
        finally:
            delivery.stop()
            watchdog.stop()
        if errors:
            raise errors[0]

    @staticmethod
    def _choose_visible_file(path: Path, evidence: Path | None) -> None:
        """Capture and fill the active external chooser using real input."""
        # Flush already-posted focus events so Ctrl+A reaches the filename field.
        QApplication.processEvents()
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QFileDialog):
            raise LookupError("No visible Qt file chooser")
        chooser = ComponentDriver(dialog)
        if not ComponentDriver._on_screen(dialog):
            raise LookupError("Qt file chooser is not on screen")
        if evidence is not None:
            # This callback is executing, not queued behind the screenshot flush,
            # so capture occurs while the chooser is still awaiting input.
            chooser.screenshot(evidence)
        filename = chooser._find(None, "fileNameEdit", editable=True)
        mouse_click(filename, filename.rect().center())
        key_click(filename, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        # Let QFileDialog's filesystem/completion events run between typed keys.
        for char in str(path):
            type_text(filename, char)
            QApplication.processEvents()
        # Completion popups consume Enter and may replace the typed path.
        if (popup := QApplication.activePopupWidget()) is not None:
            key_click(popup, Qt.Key.Key_Escape)
        chooser.click("&Open")

    def text_input(self, *, object_name: str) -> TextInputObservation:
        """Observe a named line edit; hidden controls expose no text or hints.

        Raise TypeError for a control that is not a QLineEdit.
        """
        widget = self._find(None, object_name, editable=True)
        if not isinstance(widget, QLineEdit):
            raise TypeError("Expected a line edit")
        visible = self._on_screen(widget)
        return TextInputObservation(
            text=widget.text() if visible else "",
            focused=visible and widget.hasFocus(),
            read_only=widget.isReadOnly(),
            description=widget.accessibleDescription() if visible else "",
        )

    def enter_text(
        self,
        text: str | None = None,
        value: str = "",
        *,
        object_name: str | None = None,
        commit: bool = True,
    ) -> None:
        """Replace a visible line edit; commit with Enter unless commit=False."""
        widget = self._find(text, object_name, editable=True)
        if self._can_interact(widget):
            mouse_click(widget, widget.rect().center())
            key_click(widget, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            if value == "":
                key_click(widget, Qt.Key.Key_Backspace)
            else:
                type_text(widget, value)
            if commit:
                key_click(widget, Qt.Key.Key_Return)
        QApplication.processEvents()

    def press_key(
        self,
        key: Qt.Key,
        text: str | None = None,
        *,
        object_name: str | None = None,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    ) -> None:
        """Send a real key with modifiers without assigning keyboard focus.

        With no selector, use the window's actual Qt focus, or the window when
        no child has focus. Otherwise, supply either text (a visible field label
        or caption) or object_name (an existing Qt objectName) to target a control.
        Tab uses Qt's normal focus traversal. Hidden, disabled and fully clipped
        targets are not driven. Process pending Qt events after the key.

        Raise ValueError if both selectors are supplied, or LookupError if a
        selector does not identify exactly one control.
        """
        widget = (
            self._window.focusWidget() or self._window
            if text is None and object_name is None
            else self._field_control(text, object_name)
        )
        if self._can_interact(widget):
            key_click(widget, key, modifiers)
        QApplication.processEvents()

    def select_combo(self, value: str, *, object_name: str) -> None:
        """Select an exact combo-box option using keyboard events, if usable."""
        widget = self._find(None, object_name)
        if not isinstance(widget, QComboBox):
            raise TypeError(f"{object_name!r} is not a QComboBox")
        index = widget.findText(value)
        if index < 0:
            raise LookupError(f"{object_name!r} has no option {value!r}")
        model = widget.model()
        if model is None:
            raise RuntimeError("Combo box has no option model")
        selectable = [
            row
            for row in range(widget.count())
            if model.flags(model.index(row, widget.modelColumn()))
            & Qt.ItemFlag.ItemIsEnabled
            and model.flags(model.index(row, widget.modelColumn()))
            & Qt.ItemFlag.ItemIsSelectable
        ]
        if index not in selectable:
            raise ValueError(f"{value!r} is not selectable")
        if self._can_interact(widget):
            mouse_click(widget, widget.rect().center())
            popup = widget.view()
            assert popup is not None
            # QComboBox Home skips disabled headings; the popup view's Home does not.
            key_click(widget, Qt.Key.Key_Home)
            for _ in range(selectable.index(index)):
                key_click(popup, Qt.Key.Key_Down)
            key_click(popup, Qt.Key.Key_Return)
        QApplication.processEvents()

    def _tab_bar(self, caption: str) -> QTabBar:
        bars = [
            bar
            for bar in self._window.findChildren(QTabBar)
            if self._on_screen(bar) and caption in self._tab_captions(bar)
        ]
        return _unique(caption, bars, [repr(self._tab_captions(bar)) for bar in bars])

    @staticmethod
    def _tab_captions(bar: QTabBar) -> tuple[str, ...]:
        return tuple(bar.tabText(index) for index in range(bar.count()))

    def select_tab(self, caption: str) -> None:
        """Click a uniquely named tab on a visible tab bar; ambiguity fails fast."""
        bar = self._tab_bar(caption)
        indices = [i for i in range(bar.count()) if bar.tabText(i) == caption]
        index = _unique(caption, indices, [str(i) for i in indices])
        if (
            self._can_interact(bar)
            and bar.isTabEnabled(index)
            and bar.isTabVisible(index)
        ):
            mouse_click(bar, bar.tabRect(index).center())
        QApplication.processEvents()

    def tabs(self, caption: str) -> TabObservation:
        """Read the visible tab bar containing caption, without returning Qt objects."""
        bar = self._tab_bar(caption)
        return TabObservation(bar.tabText(bar.currentIndex()), self._tab_captions(bar))

    def status(self) -> StatusObservation:
        """Read the one visible status bar's message or selectable direct labels."""
        bars = [
            bar for bar in self._window.findChildren(QStatusBar) if self._on_screen(bar)
        ]
        bar = _unique("status bar", bars, [bar.objectName() for bar in bars])
        message = bar.currentMessage()
        if not message:
            labels = bar.findChildren(
                QLabel, options=Qt.FindChildOption.FindDirectChildrenOnly
            )
            message = "\n".join(
                label.text() for label in labels if self._on_screen(label)
            )
        return StatusObservation(message)

    def dialogs(self) -> tuple[DialogObservation, ...]:
        """List visible descendant dialogs and their visible, nonempty widget text."""
        return tuple(
            DialogObservation(
                dialog.windowTitle(),
                tuple(
                    text
                    for widget in dialog.findChildren(QWidget)
                    if self._on_screen(widget) and (text := _display_text(widget))
                ),
                self._dialog_frame_within_screen(dialog),
                self._dialog_height_fraction(dialog),
            )
            for dialog in self._window.findChildren(QDialog)
            if self._on_screen(dialog)
        )

    @staticmethod
    def _dialog_frame_within_screen(dialog: QDialog) -> bool:
        screen = dialog.screen()
        if screen is None:
            raise RuntimeError("Visible dialog has no screen")
        return screen.availableGeometry().contains(dialog.frameGeometry())

    @staticmethod
    def _dialog_height_fraction(dialog: QDialog) -> float:
        screen = dialog.screen()
        if screen is None:
            raise RuntimeError("Visible dialog has no screen")
        return dialog.height() / screen.availableGeometry().height()

    def scroll_area(self, *, object_name: str) -> ScrollObservation:
        """Read a scroll area's visible scrollbar; wrong widget types fail fast."""
        widget = self._find(None, object_name)
        if not isinstance(widget, QScrollArea):
            raise TypeError(f"{object_name!r} is not a QScrollArea")
        bar = widget.verticalScrollBar()
        return ScrollObservation(
            bar is not None and self._on_screen(bar) and bar.maximum() > bar.minimum()
        )

    def legends(self) -> LegendObservation:
        """Read visible artist text only from visible matplotlib figure canvases."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        texts: list[str] = []
        for canvas in self._window.findChildren(FigureCanvasQTAgg):
            if not self._on_screen(canvas):
                continue
            for axes in canvas.figure.axes:
                legend = axes.get_legend()
                if axes.get_visible() and legend is not None and legend.get_visible():
                    texts.extend(
                        t.get_text() for t in legend.get_texts() if t.get_visible()
                    )
        return LegendObservation(tuple(texts))

    def plots(self) -> PlotObservation:
        """Read labels and visible stroked curves from shown matplotlib canvases.

        Hidden axes/canvases, transparent lines, empty paths and paths outside
        the displayed axes do not count. Counts describe public rendered artists,
        not acquisition results; use a PNG to review appearance and occlusion.
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        observed: list[PlotAxesObservation] = []
        for canvas in self._window.findChildren(FigureCanvasQTAgg):
            if not self._on_screen(canvas):
                continue
            for axes in canvas.figure.axes:
                if not axes.get_visible():
                    continue
                curves = sum(
                    1
                    for line in axes.lines
                    if line.get_visible()
                    and line.get_alpha() != 0
                    and line.get_linewidth() > 0
                    and line.get_linestyle() not in ("None", "", " ")
                    and line.get_path()
                    .transformed(line.get_transform())
                    .intersects_bbox(axes.bbox, filled=False)
                )
                observed.append(
                    PlotAxesObservation(
                        axes.get_xlabel()
                        if axes.xaxis.get_visible() and axes.xaxis.label.get_visible()
                        else "",
                        axes.get_ylabel()
                        if axes.yaxis.get_visible() and axes.yaxis.label.get_visible()
                        else "",
                        curves,
                    )
                )
        return PlotObservation(tuple(observed))

    def plot_label_bounds(self) -> tuple[PlotLabelBoundsObservation, ...]:
        """Read renderer bounds for nonempty visible ylabels on shown canvases.

        Call after a screenshot or Qt paint has rendered the current geometry.
        Coordinates share the canvas renderer's pixel space, not screen pixels.
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        observed: list[PlotLabelBoundsObservation] = []
        for canvas in self._window.findChildren(FigureCanvasQTAgg):
            if not self._on_screen(canvas):
                continue
            renderer = canvas.get_renderer()
            for axes in canvas.figure.axes:
                label = axes.yaxis.label
                if (
                    axes.get_visible()
                    and axes.yaxis.get_visible()
                    and label.get_visible()
                    and label.get_text()
                ):
                    bounds = label.get_window_extent(renderer)
                    observed.append(
                        PlotLabelBoundsObservation(
                            canvas.objectName(),
                            float(canvas.figure.bbox.x0),
                            float(axes.bbox.x0),
                            float(bounds.x0),
                            float(bounds.x1),
                            float(canvas.figure.bbox.x1),
                        )
                    )
        return tuple(observed)

    def screenshot(self, path: Path) -> Path:
        """Grab the window to caller-chosen PNG path, creating parent directories.

        Visible dialogs are grabbed separately beside it as ``<stem>-dialog-N.png``
        because top-level dialog windows are not included in QWidget.grab().
        Owned top-level main windows are saved as ``<stem>-window-N.png``.
        The active visible popup belonging to this window is saved separately as
        ``<stem>-popup.png``. Other windows' popups are never captured.
        Raises OSError if Qt cannot save any image.
        """
        QApplication.processEvents()
        path.parent.mkdir(parents=True, exist_ok=True)
        widgets = [self._window, *self._window.findChildren(QDialog)]
        for index, widget in enumerate(w for w in widgets if w.isVisible()):
            output = (
                path if index == 0 else path.with_stem(f"{path.stem}-dialog-{index}")
            )
            if not widget.grab().save(str(output), "PNG"):
                raise OSError(f"Could not save screenshot: {output}")
        for index, window in enumerate(self._window.findChildren(QMainWindow), 1):
            if window.isVisible() and window.isWindow():
                output = path.with_stem(f"{path.stem}-window-{index}")
                if not window.grab().save(str(output), "PNG"):
                    raise OSError(f"Could not save screenshot: {output}")
        popup = QApplication.activePopupWidget()
        if popup is not None and popup.isVisible():
            # isAncestorOf excludes child windows, so follow Qt's public parent chain.
            parent = popup.parentWidget()
            while parent is not None and parent is not self._window:
                parent = parent.parentWidget()
            if parent is self._window:
                output = path.with_stem(f"{path.stem}-popup")
                if not popup.grab().save(str(output), "PNG"):
                    raise OSError(f"Could not save screenshot: {output}")
        return path
