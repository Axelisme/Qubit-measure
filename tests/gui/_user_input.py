"""Deliver keyboard and mouse input to one widget, as a user's would arrive.

QTest's PyQt6 stubs read its functions as unbound instance methods, so GUI
tests send the same Qt events through QApplication.sendEvent instead.
"""

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QStatusBar,
    QStyle,
    QStyleOptionButton,
    QTabBar,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)


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


@dataclass(frozen=True)
class WidgetObservation:
    """Displayed text, effective availability, identity and hover help of a widget."""

    kind: str  # Qt widget class name, e.g. QPushButton or QLabel.
    text: str  # Rendered widget value/caption; empty when hidden or without text.
    enabled: bool  # Effective Qt enabled flag, including disabled ancestors.
    visible: bool  # Effective Qt visibility, including hidden ancestors.
    object_name: str  # Stable objectName if provided by the UI, otherwise empty.
    tooltip: str  # Hover help; empty when hidden or without a tooltip.
    checked: (
        bool | None
    )  # Visible checkable button state; None if hidden or not checkable.


@dataclass(frozen=True)
class TreeRowObservation:
    """One on-screen tree row, without model data or retained Qt objects."""

    tree_name: str  # Containing tree's objectName; empty if the UI provides none.
    path: tuple[str, ...]  # Root-to-row captions from column zero, not domain keys.
    columns: tuple[
        str, ...
    ]  # Displayed column texts in left-to-right order, no units converted.
    expanded: bool  # Whether the row currently exposes its children.


@dataclass(frozen=True)
class VisibleObservation:
    """Visible widgets only, each described without retaining a Qt object."""

    widgets: tuple[WidgetObservation, ...]
    tree_rows: tuple[
        TreeRowObservation, ...
    ]  # Rows intersecting visible tree viewports.


@dataclass(frozen=True)
class StatusObservation:
    """Text currently visible in the window's bottom status bar."""

    text: str


@dataclass(frozen=True)
class DialogObservation:
    """A visible dialog's title and visible widget texts, in widget-tree order."""

    title: str
    texts: tuple[str, ...]


@dataclass(frozen=True)
class TabObservation:
    """Selected tab caption and all captions in the matched visible tab bar."""

    current: str
    captions: tuple[str, ...]


@dataclass(frozen=True)
class LegendObservation:
    """Visible legend captions from visible matplotlib canvases, in draw order."""

    texts: tuple[str, ...]


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
    ``object_name`` instead matches exact objectName, including hidden widgets,
    so availability can be observed. Zero or multiple matches raise LookupError
    with candidate descriptions; supply exactly one selector. Operations on
    disabled, hidden, or modal-blocked widgets are no-ops. Clicks and edits use
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
                matches = widget.isVisible() and caption == text
            if matches:
                candidates.append(widget)
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
    def _can_interact(widget: QWidget) -> bool:
        modal = QApplication.activeModalWidget()
        return (
            widget.isVisible()
            and widget.isEnabled()
            and (modal is None or widget is modal or modal.isAncestorOf(widget))
        )

    def observe(
        self, text: str | None = None, *, object_name: str | None = None
    ) -> WidgetObservation:
        """Read one widget's display/availability/help; lookup errors fail fast."""
        widget = self._find(text, object_name)
        return self._observe_widget(widget)

    @staticmethod
    def _observe_widget(widget: QWidget) -> WidgetObservation:
        return WidgetObservation(
            kind=type(widget).__name__,
            text=_display_text(widget) if widget.isVisible() else "",
            enabled=widget.isEnabled(),
            visible=widget.isVisible(),
            object_name=widget.objectName(),
            tooltip=widget.toolTip() if widget.isVisible() else "",
            checked=(
                widget.isChecked()
                if widget.isVisible()
                and isinstance(widget, QAbstractButton)
                and widget.isCheckable()
                else None
            ),
        )

    def visible(self) -> VisibleObservation:
        """Snapshot all visible widgets for completion predicates with changing captions."""
        return VisibleObservation(
            widgets=tuple(
                self._observe_widget(w) for w in self._widgets() if w.isVisible()
            ),
            tree_rows=tuple(
                TreeRowObservation(
                    tree.objectName(),
                    path,
                    tuple(item.text(i) for i in range(tree.columnCount())),
                    item.isExpanded(),
                )
                for tree in self._window.findChildren(QTreeWidget)
                if tree.isVisible()
                for item, path in self._visible_tree_items(tree)
            ),
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
            path = (*parent, item.text(0))
            if viewport.rect().intersects(tree.visualItemRect(item)):
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
            [
                repr(tuple(i.text(c) for c in range(tree.columnCount())))
                for i in candidates
            ],
        )
        if not item.isExpanded() and item.flags() & Qt.ItemFlag.ItemIsEnabled:
            rect = tree.visualItemRect(item)
            viewport = tree.viewport()
            assert viewport is not None
            mouse_click(viewport, QPoint(rect.left() + 10, rect.center().y()))
            key_click(tree, Qt.Key.Key_Right)
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

    def enter_text(
        self,
        text: str | None = None,
        value: str = "",
        *,
        object_name: str | None = None,
    ) -> None:
        """Replace a line edit using its visible form label/name, then press Enter."""
        widget = self._find(text, object_name, editable=True)
        if self._can_interact(widget):
            mouse_click(widget, widget.rect().center())
            key_click(widget, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            type_text(widget, value)
            key_click(widget, Qt.Key.Key_Return)
        QApplication.processEvents()

    def _tab_bar(self, caption: str) -> QTabBar:
        bars = [
            bar
            for bar in self._window.findChildren(QTabBar)
            if bar.isVisible() and caption in self._tab_captions(bar)
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
        bars = [bar for bar in self._window.findChildren(QStatusBar) if bar.isVisible()]
        bar = _unique("status bar", bars, [bar.objectName() for bar in bars])
        message = bar.currentMessage()
        if not message:
            labels = bar.findChildren(
                QLabel, options=Qt.FindChildOption.FindDirectChildrenOnly
            )
            message = "\n".join(label.text() for label in labels if label.isVisible())
        return StatusObservation(message)

    def dialogs(self) -> tuple[DialogObservation, ...]:
        """List visible descendant dialogs and their visible, nonempty widget text."""
        return tuple(
            DialogObservation(
                dialog.windowTitle(),
                tuple(
                    text
                    for widget in dialog.findChildren(QWidget)
                    if widget.isVisible() and (text := _display_text(widget))
                ),
            )
            for dialog in self._window.findChildren(QDialog)
            if dialog.isVisible()
        )

    def legends(self) -> LegendObservation:
        """Read visible artist text only from visible matplotlib figure canvases."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        texts: list[str] = []
        for canvas in self._window.findChildren(FigureCanvasQTAgg):
            if not canvas.isVisible():
                continue
            for axes in canvas.figure.axes:
                legend = axes.get_legend()
                if legend is not None and legend.get_visible():
                    texts.extend(
                        t.get_text() for t in legend.get_texts() if t.get_visible()
                    )
        return LegendObservation(tuple(texts))

    def screenshot(self, path: Path) -> Path:
        """Grab the window to caller-chosen PNG path, creating parent directories.

        Visible dialogs are grabbed separately beside it as ``<stem>-dialog-N.png``
        because top-level dialog windows are not included in QWidget.grab().
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
        return path
