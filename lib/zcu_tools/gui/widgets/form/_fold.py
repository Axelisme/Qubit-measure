"""Fold titles and nested row layout, shared by field and group sections."""

from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QSignalBlocker, QSize, Qt
from qtpy.QtGui import QKeyEvent, QMouseEvent, QPaintEvent
from qtpy.QtWidgets import (
    QApplication,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QStyle,
    QStyleOptionFocusRect,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

# Shared vertical spacing for a form's root and nested rows.
ROW_SPACING = 4
_GROUP_MARGINS = (8, 4, 4, 4)
# A fold title starts with one of these marks, a space, then its label.
FOLDED_MARK = "\u25b8"
EXPANDED_MARK = "\u25be"
# Keys that fold or unfold a fold title that has keyboard focus.
_FOLD_KEYS = frozenset({Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter})


class FoldTitle(QLabel):
    """A bold section title that folds its body on a click or a key.

    Its text is FOLDED_MARK while body is hidden, EXPANDED_MARK while body
    is shown, then a space and label. A press of the left button flips
    body's visibility and the mark, then calls toggled(). Tab (not a click)
    gives the title keyboard focus; while it has focus the style's focus
    frame is drawn around it, and Space, Return or Enter flips it the same
    way. Other keys do nothing, and keys typed into other widgets never
    reach it. It is a title, not an input, so a read-only form still folds;
    a disabled one gets no mouse or key input.
    A QLabel that overrides the mouse, key and paint events is the plainest
    Qt widget that reads as a title and reacts to a click.
    """

    fold_changed = pyqtSignal()

    def __init__(
        self,
        label: str,
        body: QWidget,
        parent: QWidget,
        toggled: Callable[[], None],
    ) -> None:
        """Create a title for body under parent, calling toggled after a fold.

        label is plain display text; body is an existing content widget whose
        visibility this title controls. Construction reads body visibility but
        does not change it or call toggled. User folds call toggled with no
        arguments, then emit fold_changed. All operations run on the Qt thread.
        """
        super().__init__(parent)
        self._label = label
        self._body = body
        self._toggled = toggled
        self._foldable = True
        font = self.font()
        font.setBold(True)
        self.setFont(font)
        # Never take spare width, so controls sit beside the title.
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._show_mark()

    def sizeHint(self) -> QSize:
        """QLabel's hint, widened to fit the wider mark.

        A label column gets the fixed width of its widest hint at build;
        fitting both marks keeps a flipped title from clipping its text.
        """
        hint = super().sizeHint()
        metrics = self.fontMetrics()
        widest = max(
            metrics.horizontalAdvance(mark) for mark in (FOLDED_MARK, EXPANDED_MARK)
        )
        extra = widest - metrics.horizontalAdvance(self._mark())
        return QSize(hint.width() + extra, hint.height())

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        """Fold or unfold the body on a left-button press ev."""
        if ev is None or ev.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(ev)
            return
        ev.accept()
        self._flip()

    def keyPressEvent(self, ev: QKeyEvent | None) -> None:
        """Fold or unfold the body on a Space, Return or Enter press ev."""
        if ev is None or ev.key() not in _FOLD_KEYS:
            super().keyPressEvent(ev)
            return
        ev.accept()
        self._flip()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Paint the title, then the style's focus frame while it has focus."""
        super().paintEvent(a0)
        if not self.hasFocus():
            return
        option = QStyleOptionFocusRect()
        option.initFrom(self)
        painter = QStylePainter(self)
        painter.drawPrimitive(QStyle.PrimitiveElement.PE_FrameFocusRect, option)
        painter.end()

    def set_foldable(self, *, available: bool, expanded: bool) -> None:
        """Show a fold mark only when available; otherwise hide the empty body.

        expanded sets the body's visibility when a previously empty layer
        gains children. This never emits fold_changed or writes form state.
        """
        if self._foldable == available:
            return
        self._foldable = available
        font = self.font()
        font.setBold(available)
        self.setFont(font)
        self._body.setVisible(available and expanded)
        self._show_mark()

    def _flip(self) -> None:
        if not self._foldable:
            return
        focused = QApplication.focusWidget()
        if focused is not None and self._body.isAncestorOf(focused):
            # Hiding a focused text field emits editingFinished. Folding is
            # navigation, not a commit: keep its widget-local text untouched.
            with QSignalBlocker(focused):
                self._body.hide()
        else:
            self._body.setVisible(self._body.isHidden())
        self._show_mark()
        self._toggled()
        self.fold_changed.emit()

    def _mark(self) -> str:
        return FOLDED_MARK if self._body.isHidden() else EXPANDED_MARK

    def _show_mark(self) -> None:
        self.setText(f"{self._mark()} {self._label}" if self._foldable else self._label)


def foldable_rows[K](
    parent: QWidget,
    folds: dict[K, bool],
    key: K,
    *,
    label: str,
    name: str,
    expanded: bool,
) -> tuple[FoldTitle, QGroupBox, QVBoxLayout]:
    """Build a fold title and the foldable box of rows it folds, under parent.

    Returns the title, named name and showing label after its fold mark;
    the box; and the box's rows layout, which has nested-group margins and
    ROW_SPACING. The caller puts the title in a header and the box below
    it. folds records user folds (True is expanded) across rebuilds: the
    box starts shown when folds[key] is True; without that key, when
    expanded is True. Each click on the title flips the box's visibility
    and the title's mark and records the new state as folds[key]. The title's
    fold_changed signal lets callers build the first expanded layer.
    """
    body = QGroupBox(parent)
    rows = QVBoxLayout(body)
    rows.setContentsMargins(*_GROUP_MARGINS)
    rows.setSpacing(ROW_SPACING)
    body.setVisible(folds.get(key, expanded))

    def record() -> None:
        folds[key] = not body.isHidden()

    title = FoldTitle(label, body, parent, record)
    title.setObjectName(name)
    return title, body, rows
