"""Select controls and apply user-facing component input policy."""

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QEvent, QPoint, QPointF, QSize, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QCalendarWidget,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QScrollArea,
    QSplitter,
    QStyle,
    QStyleOptionButton,
    QTabBar,
    QTableView,
    QTableWidget,
    QTreeWidget,
    QWidget,
)

from .._observations import (
    DialogObservation,
    LegendObservation,
    ListRowObservation,
    MenuActionObservation,
    PlotLabelBoundsObservation,
    PlotObservation,
    ScrollObservation,
    StatusObservation,
    TableRowObservation,
    TabObservation,
    TextInputObservation,
    VisibleObservation,
    WidgetObservation,
)
from . import collections, input, menus, observing
from .input import key_click, mouse_click, mouse_double_click, type_text
from .selection import can_interact, display_text, on_screen, unique


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
                    self._field_label(widget) if editable else display_text(widget)
                )
                matches = on_screen(widget) and caption == text
            if matches:
                candidates.append(widget)
        if object_name is not None:
            visible_matches = [widget for widget in candidates if on_screen(widget)]
            if visible_matches:
                candidates = visible_matches
        descriptions = [
            f"{type(w).__name__}(objectName={w.objectName()!r}, text={display_text(w)!r})"
            for w in candidates
        ]
        return unique(text or object_name or "", candidates, descriptions)

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

    def resize_window(self, width: int, height: int) -> None:
        """Resize the shown window in logical pixels and flush its Qt layout.

        Hidden or modal-blocked windows are not resized. Nonpositive dimensions
        raise ValueError. Qt may enforce the window's minimum size.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Window dimensions must be positive")
        if can_interact(self._window):
            self._window.resize(width, height)
        QApplication.processEvents()

    def close_window(self) -> None:
        """Request the shown window's normal Qt close, then flush owner events."""
        if can_interact(self._window):
            self._window.close()
        QApplication.processEvents()

    def click_at(self, x: int, y: int) -> None:
        """Click global logical coordinates through the window's actual hit target.

        Coordinates outside this shown window are a no-op. Child overlays receive
        the event instead of covered controls, just as with physical mouse input.
        """
        point = self._window.mapFromGlobal(QPoint(x, y))
        if can_interact(self._window) and self._window.rect().contains(point):
            target = self._window.childAt(point) or self._window
            if can_interact(target):
                mouse_click(target, target.mapFromGlobal(QPoint(x, y)))
        QApplication.processEvents()

    def drag_at(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Drag between global logical points on one visible, enabled hit target.

        Both endpoints must remain in that target. Covered or clipped targets
        are not driven. Delivers a left press, move and release, then flushes Qt.
        """
        target = self._window.childAt(self._window.mapFromGlobal(QPoint(*start)))
        if target is None or not can_interact(target):
            return
        if not target.visibleRegion().contains(target.mapFromGlobal(QPoint(*end))):
            return
        for kind, at, button, buttons in (
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
            screen = QPoint(*at)
            QApplication.sendEvent(
                target,
                QMouseEvent(
                    kind,
                    QPointF(target.mapFromGlobal(screen)),
                    QPointF(screen),
                    button,
                    buttons,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
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
        return observing.observe_widget(widget)

    def image_center_colour(self, *, object_name: str) -> str:
        """Read the displayed QLabel raster's centre pixel as a hex RGB colour.

        Hidden/clipped labels raise LookupError; a label without a displayed
        raster raises ValueError. Other widget kinds raise TypeError.
        """
        widget = self._find(None, object_name)
        if not on_screen(widget):
            raise LookupError("Raster is not visible")
        if not isinstance(widget, QLabel):
            raise TypeError("Expected a raster label")
        pixmap = widget.pixmap()
        if pixmap.isNull():
            raise ValueError("Label has no displayed raster")
        return (
            pixmap.toImage()
            .pixelColor(pixmap.width() // 2, pixmap.height() // 2)
            .name()
        )

    def list_rows(self) -> tuple[ListRowObservation, ...]:
        """Observe rows intersecting visible QListWidget viewports in this window."""

        return observing.list_rows(self._window)

    def select_list_row(self, text: str) -> None:
        """Click the unique visible row with this exact rendered caption."""
        listing, item = self._visible_list_row(text)
        viewport = listing.viewport()
        assert viewport is not None
        mouse_click(viewport, listing.visualItemRect(item).center())

    def double_click_list_row(self, text: str) -> None:
        """Double-click the unique visible, enabled list row with exact caption.

        Raise LookupError for missing, ambiguous or clipped rows. Send real mouse
        events to the viewport and process their resulting Qt events.
        """
        listing, item = self._visible_list_row(text)
        viewport = listing.viewport()
        assert viewport is not None
        at = listing.visualItemRect(item).center()
        mouse_double_click(viewport, at)
        QApplication.processEvents()

    def _visible_list_row(self, text: str) -> tuple[QListWidget, QListWidgetItem]:
        matches: list[tuple[QListWidget, QListWidgetItem]] = []
        for listing in self._window.findChildren(QListWidget):
            if not can_interact(listing):
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
        return unique(text, matches, [i.text() for _, i in matches])

    def table_rows(self, *, object_name: str) -> tuple[TableRowObservation, ...]:
        """Observe rows intersecting the named QTableWidget viewport."""
        table = self._find(None, object_name)
        if not isinstance(table, QTableWidget):
            raise TypeError(f"{object_name!r} is not a QTableWidget")
        if not on_screen(table):
            return ()
        return collections.table_rows(table)

    def select_table_row(
        self, text: str, *, object_name: str, double_click: bool = False
    ) -> None:
        """Click the unique fully visible table cell with this exact caption.

        Disabled, hidden, clipped and modal-blocked tables cannot be clicked.
        The target must already be in the viewport; this does not auto-scroll.
        double_click sends the normal press/release followed by a double-click event.
        """
        table = self._find(None, object_name)
        if not isinstance(table, QTableWidget):
            raise TypeError(f"{object_name!r} is not a QTableWidget")
        if not can_interact(table):
            return
        collections.select_table_row(table, text, double_click=double_click)

    def drag_column(self, column: int, delta: int, *, object_name: str) -> None:
        """Drag a visible table/tree header's right boundary by delta pixels.

        column is a zero-based logical column. Hidden, disabled or clipped
        boundaries are not driven. A non-table selector raises TypeError;
        out-of-range columns raise IndexError. Resize mode is left unchanged.
        """
        table = self._find(None, object_name)
        if not isinstance(table, (QTableWidget, QTreeWidget)):
            raise TypeError(f"{object_name!r} is not a table or tree")
        collections.drag_column(table, column, delta)

    def edit_table_value(
        self, current: str, value: str, *, object_name: str, column: str
    ) -> None:
        """Edit the matching caption under a public header, using real keys."""
        table = self._find(None, object_name)
        if not isinstance(table, QTableWidget):
            raise TypeError(f"{object_name!r} is not a QTableWidget")
        if not can_interact(table):
            return
        columns = [
            index
            for index in range(table.columnCount())
            if (header := table.horizontalHeaderItem(index)) is not None
            and header.text() == column
        ]
        index = unique(column, columns, [str(index) for index in columns])
        viewport = table.viewport()
        assert viewport is not None
        items = [
            item
            for row in range(table.rowCount())
            if (item := table.item(row, index)) is not None
            and item.text() == current
            and viewport.visibleRegion().contains(table.visualItemRect(item))
        ]
        item = unique(current, items, [item.text() for item in items])
        mouse_click(viewport, table.visualItemRect(item).center())
        self.press_key(Qt.Key.Key_F2, object_name=object_name)
        widget = self._window.focusWidget()
        if not isinstance(widget, QLineEdit) or not can_interact(widget):
            raise RuntimeError("Table cell did not open a visible text editor")
        key_click(widget, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        type_text(widget, value)
        key_click(widget, Qt.Key.Key_Return)
        QApplication.processEvents()

    def visible(self) -> VisibleObservation:
        """Snapshot all visible widgets for completion predicates with changing captions."""

        return observing.visible(self._window)

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
        if not can_interact(tree):
            return
        collections.expand_tree(tree, path)

    def select_tree_row(self, path: tuple[str, ...], *, object_name: str) -> None:
        """Click a visible tree row's first column, giving it keyboard focus.

        Exact visible captions identify the path. Ancestors must already be
        expanded; disabled, hidden and viewport-clipped targets are not driven.
        Missing or ambiguous visible paths raise LookupError.
        """
        tree = self._find(None, object_name)
        if not isinstance(tree, QTreeWidget):
            raise TypeError(f"{object_name!r} is not a QTreeWidget")
        if not can_interact(tree):
            return
        collections.select_tree_row(tree, path)

    def edit_tree_value(
        self,
        path: tuple[str, ...],
        *,
        object_name: str,
        value: str,
        commit: bool = True,
    ) -> None:
        """Edit a visible tree row's Value cell using double-click and optional Enter.

        commit=False leaves the editor open for a later Enter or focus loss.

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
        if not can_interact(tree):
            return
        collections.edit_tree_value(tree, path, value=value, commit=commit)

    def click(
        self,
        text: str | None = None,
        *,
        object_name: str | None = None,
        double_click: bool = False,
    ) -> None:
        """Click a usable widget with real mouse events.

        Checkboxes use the style's indicator centre: a stretched checkbox's
        widget centre can be outside Qt's clickable label/indicator region.
        Other widgets use their rectangle centre. Availability rules are unchanged.
        """
        widget = self._find(text, object_name)
        if can_interact(widget):
            if isinstance(widget, QDateTimeEdit):
                editor = widget.findChild(QLineEdit)
                widget = editor
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
            if double_click:
                mouse_double_click(widget, point)
            else:
                mouse_click(widget, point)
        QApplication.processEvents()

    def drag_splitter(self, delta: int, *, object_name: str) -> None:
        """Drag the first visible splitter handle by delta pixels along its axis.

        The splitter must have two or more children. Hidden/disabled handles and
        clipped drag endpoints are not driven. Non-splitters raise TypeError.
        """
        splitter = self._find(None, object_name)
        if not isinstance(splitter, QSplitter):
            raise TypeError(f"{object_name!r} is not a splitter")
        input.drag_splitter(splitter, delta)

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
            if on_screen(widget) and widget.toolTip() == tooltip
        ]
        widget = unique(tooltip, candidates, [w.objectName() for w in candidates])
        if can_interact(widget):
            mouse_click(widget, widget.rect().center())
        QApplication.processEvents()

    def _field_control(self, text: str | None, object_name: str | None) -> QWidget:
        if (text is None) == (object_name is None):
            raise ValueError("Supply exactly one of text or object_name")
        if text is not None:
            fields = [
                widget
                for widget in self._widgets()
                if on_screen(widget) and self._field_label(widget) == text
            ]
            if fields:
                return unique(text, fields, [widget.objectName() for widget in fields])
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
        if not can_interact(widget):
            return
        input.choose_option(widget, value)

    def invoke_context_action(
        self,
        text: str | None = None,
        action: str = "",
        *,
        object_name: str | None = None,
        evidence: Path | None = None,
        tree_path: tuple[str, ...] | None = None,
    ) -> None:
        """Open a control's context menu and click an exact visible action caption.

        Supply exactly one exact text/form-label or object_name selector. Missing/
        ambiguous captions raise LookupError. Unavailable widgets and disabled actions are no-ops. Hidden
        actions are not selected. Real context-menu/mouse events are delivered;
        actions and product handlers are never called directly. A scheduled menu
        delivery has a 30-second watchdog; all popup/callback failures are returned
        to the caller after closing the menu, never left in Qt's event loop.
        evidence optionally captures the visible popup before selecting the action.
        tree_path targets a visible QTreeWidget row instead of the control centre;
        missing/ambiguous paths raise LookupError and non-tree targets TypeError.
        """
        widget = self._field_control(text, object_name)
        if not can_interact(widget):
            return
        target = (
            widget.viewport() if isinstance(widget, QAbstractScrollArea) else widget
        )
        assert target is not None
        point = target.rect().center()
        if tree_path is not None:
            if not isinstance(widget, QTreeWidget):
                raise TypeError("Expected a tree for tree_path")
            item = unique(
                repr(tree_path),
                [
                    i
                    for i, path in collections.visible_tree_items(widget)
                    if path == tree_path
                ],
                [repr(path) for _, path in collections.visible_tree_items(widget)],
            )
            point = QPoint(
                widget.columnViewportPosition(0) + widget.columnWidth(0) // 2,
                widget.visualItemRect(item).center().y(),
            )
        if not target.visibleRegion().contains(point):
            return
        menus.invoke_context_action(target, point, action, evidence)

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
        self._visit_menu(opener, lambda: menus.select_menu_path(path))

    def menu_actions(
        self, opener: str, *, evidence: Path | None = None
    ) -> tuple[MenuActionObservation, ...]:
        """Open a menu, snapshot its visible entries in order, then press Escape.

        Uses the same mouse delivery and watchdog as select_menu. Disabled
        openers return no entries. evidence captures the window and popup while
        expanded. Missing openers or popups raise LookupError.
        """
        entries: list[MenuActionObservation] = []

        def observe() -> None:
            menu = QApplication.activePopupWidget()
            if not isinstance(menu, QMenu):
                raise LookupError("No visible popup menu")
            entries.extend(
                MenuActionObservation(
                    a.text(), a.isEnabled(), a.objectName(), a.shortcut().toString()
                )
                for a in menu.actions()
                if a.isVisible() and not menu.actionGeometry(a).isEmpty()
            )
            if evidence is not None:
                self.screenshot(evidence)
            menus.dismiss_menus()

        self._visit_menu(opener, observe)
        return tuple(entries)

    def _visit_menu(self, opener: str, visit: Callable[[], None]) -> None:
        button = self._find(opener, None)
        if not can_interact(button):
            return
        menus.visit_menu(lambda: self.click(opener), visit)

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
        matches = [w for w in self._widgets() if display_text(w) == opener]
        visible = [w for w in matches if on_screen(w)]
        candidates = visible or matches
        button = unique(
            opener,
            candidates,
            [f"{type(w).__name__}(objectName={w.objectName()!r})" for w in candidates],
        )
        if not can_interact(button):
            return
        menus.deliver_file_choice(
            lambda: self.click(opener),
            lambda: self._choose_visible_file(path, evidence),
        )

    @staticmethod
    def _choose_visible_file(path: Path, evidence: Path | None) -> None:
        """Capture and fill the active external chooser using real input."""
        # Flush already-posted focus events so Ctrl+A reaches the filename field.
        QApplication.processEvents()
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QFileDialog):
            raise LookupError("No visible Qt file chooser")
        chooser = ComponentDriver(dialog)
        if not on_screen(dialog):
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

    def focused_input(self) -> WidgetObservation:
        """Observe the currently focused visible input without moving focus."""
        widget = self._window.focusWidget()
        if widget is None:
            raise LookupError("No focused input")
        return observing.observe_widget(widget)

    def text_input(self, *, object_name: str) -> TextInputObservation:
        """Observe a named line edit; hidden controls expose no text or hints.

        Raise TypeError for a control that is not a QLineEdit.
        """
        widget = self._find(None, object_name, editable=True)
        if not isinstance(widget, QLineEdit):
            raise TypeError("Expected a line edit")
        visible = on_screen(widget)
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
        if can_interact(widget):
            mouse_click(widget, widget.rect().center())
            key_click(widget, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            if value == "":
                key_click(widget, Qt.Key.Key_Backspace)
            else:
                type_text(widget, value)
            if commit:
                key_click(widget, Qt.Key.Key_Return)
        QApplication.processEvents()

    def type_text(self, value: str) -> None:
        """Type into the current visible, enabled keyboard focus without moving it."""
        widget = self._window.focusWidget()
        if widget is None:
            raise LookupError("No focused input")
        if can_interact(widget):
            type_text(widget, value)
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
        if can_interact(widget):
            key_click(widget, key, modifiers)
        QApplication.processEvents()

    def open_calendar(self, *, object_name: str) -> None:
        """Click an enabled date edit's visible right-side calendar arrow.

        A date edit without calendarPopup raises ValueError. Other selectors
        raise TypeError. Hidden, disabled or clipped edits are not driven.
        """
        widget = self._find(None, object_name)
        if not isinstance(widget, QDateTimeEdit):
            raise TypeError(f"{object_name!r} is not a date edit")
        if not widget.calendarPopup():
            raise ValueError("Date edit has no calendar popup")
        if can_interact(widget):
            mouse_click(widget, QPoint(widget.width() - 8, widget.height() // 2))
        QApplication.processEvents()

    def choose_calendar_day(self, day: int) -> None:
        """Click a unique visible day number in the active calendar popup.

        day belongs to the displayed month. Missing/ambiguous visible day cells
        raise LookupError; no date or model state is assigned directly.
        """
        popup = QApplication.activePopupWidget()
        if popup is None:
            raise LookupError("No visible calendar popup")
        calendar = unique("calendar", popup.findChildren(QCalendarWidget), [])
        view = unique("calendar view", calendar.findChildren(QTableView), [])
        model = view.model()
        viewport = view.viewport()
        assert model is not None and viewport is not None
        cells = [
            model.index(row, column)
            for row in range(model.rowCount())
            for column in range(model.columnCount())
            if str(model.index(row, column).data()) == str(day)
            and viewport.visibleRegion().contains(
                view.visualRect(model.index(row, column))
            )
        ]
        cell = unique(f"day {day}", cells, [])
        mouse_click(viewport, view.visualRect(cell).center())
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
        if can_interact(widget):
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
            if on_screen(bar) and caption in self._tab_captions(bar)
        ]
        return unique(caption, bars, [repr(self._tab_captions(bar)) for bar in bars])

    @staticmethod
    def _tab_captions(bar: QTabBar) -> tuple[str, ...]:
        return tuple(bar.tabText(index) for index in range(bar.count()))

    def select_tab(self, caption: str) -> None:
        """Click a uniquely named tab on a visible tab bar; ambiguity fails fast."""
        bar = self._tab_bar(caption)
        indices = [i for i in range(bar.count()) if bar.tabText(i) == caption]
        index = unique(caption, indices, [str(i) for i in indices])
        if can_interact(bar) and bar.isTabEnabled(index) and bar.isTabVisible(index):
            mouse_click(bar, bar.tabRect(index).center())
        QApplication.processEvents()

    def tabs(self, caption: str) -> TabObservation:
        """Read the visible tab bar containing caption, without returning Qt objects."""
        bar = self._tab_bar(caption)
        return TabObservation(bar.tabText(bar.currentIndex()), self._tab_captions(bar))

    def status(self) -> StatusObservation:
        """Read the one visible status bar's message or selectable direct labels."""

        return observing.status(self._window)

    def dialogs(self) -> tuple[DialogObservation, ...]:
        """List visible descendant dialogs and their visible, nonempty widget text."""

        return observing.dialogs(self._window)

    def scroll_area(self, *, object_name: str) -> ScrollObservation:
        """Read a scroll area's visible scrollbar; wrong widget types fail fast."""
        widget = self._find(None, object_name)
        if not isinstance(widget, QScrollArea):
            raise TypeError(f"{object_name!r} is not a QScrollArea")
        bar = widget.verticalScrollBar()
        return ScrollObservation(
            bar is not None and on_screen(bar) and bar.maximum() > bar.minimum()
        )

    def legends(self) -> LegendObservation:
        """Read visible artist text only from visible matplotlib figure canvases."""

        return observing.legends(self._window)

    def plots(self) -> PlotObservation:
        """Read labels and visible stroked curves from shown matplotlib canvases.

        Hidden axes/canvases, transparent lines, empty paths and paths outside
        the displayed axes do not count. Counts describe public rendered artists,
        not acquisition results; use a PNG to review appearance and occlusion.
        """

        return observing.plots(self._window)

    def plot_label_bounds(self) -> tuple[PlotLabelBoundsObservation, ...]:
        """Read renderer bounds for nonempty visible ylabels on shown canvases.

        Call after a screenshot or Qt paint has rendered the current geometry.
        Coordinates share the canvas renderer's pixel space, not screen pixels.
        """

        return observing.plot_label_bounds(self._window)

    def screenshot(self, path: Path) -> Path:
        """Grab the window to caller-chosen PNG path, creating parent directories.

        Visible dialogs are grabbed separately beside it as ``<stem>-dialog-N.png``
        because top-level dialog windows are not included in QWidget.grab().
        Owned top-level main windows are saved as ``<stem>-window-N.png``.
        The active visible popup belonging to this window is saved separately as
        ``<stem>-popup.png``. Other windows' popups are never captured.
        Raises OSError if Qt cannot save any image.
        """

        return observing.screenshot(self._window, path)
