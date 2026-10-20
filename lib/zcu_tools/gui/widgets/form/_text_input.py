"""Native text input commits and inline Expression completion presentation."""

import re

from qtpy.QtCore import QEvent, QPointF, QRect, QSignalBlocker, Qt
from qtpy.QtGui import QFocusEvent, QKeyEvent, QPainter, QPaintEvent, QPalette
from qtpy.QtWidgets import QLineEdit, QStyle, QStyleOptionFrame, QWidget

from zcu_tools.gui.widgets import mixed_colour


class TextInput(QLineEdit):
    """Native line edit with local inline completion and one commit on focus loss.

    set_completion_candidates installs sorted, captured $alias/$alias.field names.
    set_expression_mode enables hints for editable Expression input only, with
    no selection and whitespace or end of text immediately after the cursor.
    Paint the unmatched suffix in grey and shift displayed right-hand text. Tab accepts
    without committing or losing focus; repeated Tab cycles the original matches.
    Esc hides hints until another edit/cursor move; focus loss ends a cycle.
    accessibleDescription announces the painted suffix and keyboard controls.

    Qt clears its edited flag even when folding blocks editingFinished. The
    widget still owns the unchanged text; normal focus loss must retry it.
    Fold/retirement signal blockers suppress this notification too. Popup
    focus is temporary, so opening the text menu does not commit.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._candidates: tuple[str, ...] = ()
        self._expression_mode = False
        self._dismissed = False
        self._ghost = ""
        self._cycle: tuple[str, ...] = ()
        self._cycle_index = 0
        self._start = 0
        self._end = 0
        self.textChanged.connect(self._reset_completion)
        self.cursorPositionChanged.connect(self._reset_completion)
        self.selectionChanged.connect(self._reset_completion)

    def set_completion_candidates(self, candidates: tuple[str, ...]) -> None:
        """Use captured, sorted $alias and $alias.field names, without source reads."""
        if self._candidates != candidates:
            self._candidates = candidates
            self._reset_completion()

    def set_expression_mode(self, *, expression: bool) -> None:
        """Enable hints only for editable Expression input; never change the text."""
        self._expression_mode = expression
        self._reset_completion()

    def _matches(self) -> tuple[int, str, tuple[str, ...]]:
        """Match a one-level $token prefix before whitespace or end of text.

        Any immediate non-whitespace character, including an identifier tail,
        closing parenthesis or operator, blocks hints. Each dotted level matches
        only its own captured names. Selected text is excluded by _can_complete.
        """
        cursor = self.cursorPosition()
        text = self.text()
        if cursor < len(text) and not text[cursor].isspace():
            return 0, "", ()
        match = re.search(
            r"(?<![\w$.])\$[A-Za-z0-9_]*(?:\.[A-Za-z0-9_]*)?$", text[:cursor]
        )
        if match is None:
            return 0, "", ()
        prefix = match.group()
        matches = tuple(
            name
            for name in self._candidates
            if name.startswith(prefix)
            and len(name) > len(prefix)
            and name.count(".") == prefix.count(".")
        )
        return match.start(), prefix, matches

    def _can_complete(self) -> bool:
        return (
            self._expression_mode
            and self.hasFocus()
            and self.isEnabled()
            and not self.isReadOnly()
            and not self.hasSelectedText()
            and not self._dismissed
        )

    def _reset_completion(self) -> None:
        self._cycle = ()
        self._dismissed = False
        self._update_hint()

    def _update_hint(self) -> None:
        self._ghost = ""
        if self._can_complete() and not self._cycle:
            _, prefix, matches = self._matches()
            if matches:
                self._ghost = matches[0][len(prefix) :]
        # Screen readers receive the same suffix that paintEvent draws.
        self.setAccessibleDescription(
            f"Completion: {self._ghost}. Tab accepts; repeated Tab cycles; Esc dismisses."
            if self._ghost
            else ""
        )
        self.update()

    def _accept_completion(self) -> bool:
        """First Tab accepts; repeated Tab cycles the original sorted matches.

        Only the token is replaced. Tab does not commit or move focus while
        accepting/cycling. Any ordinary edit, cursor move or focus loss ends
        the cycle; Esc also suppresses hints until the next edit/cursor move.
        """
        if not self._can_complete():
            return False
        if self._cycle:
            self._cycle_index = (self._cycle_index + 1) % len(self._cycle)
        else:
            start, prefix, matches = self._matches()
            if not matches:
                return False
            self._start = start
            self._end = start + len(prefix)
            self._cycle = matches
            self._cycle_index = 0
        replacement = self._cycle[self._cycle_index]
        # Qt emits text/selection/cursor signals during replacement. Keep the
        # original cycle until the next real user edit, not these internal moves.
        with QSignalBlocker(self):
            self.setSelection(self._start, self._end - self._start)
            self.insert(replacement)
        self._end = self._start + len(replacement)
        self._update_hint()
        return True

    def event(self, a0: QEvent | None) -> bool:
        """Handle completion keys before QWidget turns Tab into focus traversal."""
        if isinstance(a0, QKeyEvent) and a0.type() == QEvent.Type.KeyPress:
            if (
                a0.key() == Qt.Key.Key_Tab
                and a0.modifiers() == Qt.KeyboardModifier.NoModifier
                and self._accept_completion()
            ):
                a0.accept()
                return True
            if a0.key() == Qt.Key.Key_Escape and (self._ghost or self._cycle):
                self._cycle = ()
                self._dismissed = True
                self._update_hint()
                a0.accept()
                return True
        return super().event(a0)

    def focusInEvent(self, a0: QFocusEvent | None) -> None:
        """Restore a hint on ordinary focus entry without committing input."""
        super().focusInEvent(a0)
        self._reset_completion()

    def focusOutEvent(self, a0: QFocusEvent | None) -> None:
        """Let Qt update focus styling, then notify the row once."""
        with QSignalBlocker(self):
            super().focusOutEvent(a0)
        self._reset_completion()
        if a0 is not None and a0.reason() != Qt.FocusReason.PopupFocusReason:
            self.editingFinished.emit()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Paint a grey suffix inline and shift the displayed right-hand text.

        Native QLineEdit owns the frame, scrolling, selection and caret. Only
        the area after the caret is repainted, clipped inside the native text
        rectangle. The draft and its cursor offsets remain unchanged.
        """
        super().paintEvent(a0)
        if not self._ghost:
            return
        option = QStyleOptionFrame()
        self.initStyleOption(option)
        style = self.style()
        if style is None:
            return
        content = style.subElementRect(
            QStyle.SubElement.SE_LineEditContents, option, self
        )
        margins = self.textMargins()
        content = content.adjusted(margins.left() + 2, 0, -margins.right() - 2, 0)
        x = self.cursorRect().center().x() + 1
        metrics = self.fontMetrics()
        baseline = content.center().y() + (metrics.ascent() - metrics.descent()) / 2
        painter = QPainter(self)
        painter.setClipRect(content)
        painter.fillRect(
            QRect(x, content.top(), max(0, content.right() - x + 1), content.height()),
            self.palette().brush(QPalette.ColorRole.Base),
        )
        painter.setPen(
            mixed_colour(
                self.palette().color(QPalette.ColorRole.Text),
                self.palette().color(QPalette.ColorRole.Base),
                0.5,
            )
        )
        painter.drawText(QPointF(x, baseline), self._ghost)
        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        painter.drawText(
            QPointF(x + metrics.horizontalAdvance(self._ghost), baseline),
            self.text()[self.cursorPosition() :],
        )
        painter.end()
