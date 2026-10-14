"""Qt rows and JSON text parsing, independent of form evaluation/validation."""

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

from pydantic import ConfigDict, JsonValue, TypeAdapter

# qtpy.QtGui.QAction resolves to NoReturn for pyright; PyQt6 is the binding.
from PyQt6.QtGui import QAction
from qtpy.QtCore import QEvent, QObject, QPoint, QSize, Qt
from qtpy.QtGui import QColor, QKeyEvent, QMouseEvent, QPaintEvent, QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QStyle,
    QStyleOptionFocusRect,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.cfg.codec import decode_complex, encode_complex
from zcu_tools.gui.form import FormValue, FormWritePath
from zcu_tools.gui.widgets.cfg.fields.common import (
    connect_committed_value_widget,
    make_value_widget,
    write_value_widget,
)
from zcu_tools.gui.widgets.spinbox import trimmed_decimals

from ._schema_view import FieldView, OptionView

_JSON = TypeAdapter(JsonValue, config=ConfigDict(allow_inf_nan=False))
# Vertical density: pixels between sibling rows, between a row's header and
# its error/group, and around a nested group's rows (left indents the group).
ROW_SPACING = 4
_HEADER_SPACING = 2
_GROUP_MARGINS = (8, 4, 4, 4)
# Tooltip line on rows whose input has the right-click Expression action.
_MODE_HINT = "Right-click to switch Direct / Expression"
# Share of the palette's Highlight mixed into an Expression input's Base colour:
# a light tint of the theme's own colours, so text stays readable in light
# and dark palettes alike.
_EXPRESSION_TINT = 0.18
# Shown in a group block's header while any of its rows shows an error, so a
# folded group still tells the user where Start is blocked.
GROUP_ERRORS_NOTE = "Contains errors"
# A fold title starts with one of these marks, a space, then its label.
FOLDED_MARK = "\u25b8"
EXPANDED_MARK = "\u25be"
# Suffix of the option dropdown item that shows an input no option matches.
MISSING_SUFFIX = " (missing)"
_ERRORS_NOTE_STYLE = "QLabel { color: #b3261e; }"
# User input a read-only dropdown or checkbox drops. Tab and Backtab key
# presses still pass, so focus can move on past the control.
_BLOCKED_EVENTS = frozenset(
    {
        QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonRelease,
        QEvent.Type.MouseButtonDblClick,
        QEvent.Type.KeyPress,
        QEvent.Type.KeyRelease,
    }
)
_FOCUS_KEYS = frozenset({Qt.Key.Key_Tab, Qt.Key.Key_Backtab})
# Keys that fold or unfold a fold title that has keyboard focus.
_FOLD_KEYS = frozenset({Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter})


class _UserInputBlock(QObject):
    """Event filter that keeps dropdowns and checkboxes from user edits.

    Qt has no read-only mode for QComboBox or QCheckBox, and setEnabled(False)
    greys their text. While blocking is True, the watched controls get no
    mouse clicks, no key presses except Tab and Backtab, and no wheel turns;
    a wheel turn goes on to the parents, so a scroll area around the form
    still scrolls. While it is False, every event passes.
    """

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.blocking = False

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Drop a1 for the watched control a0 while blocking; else pass it."""
        del a0  # Every watched control is treated the same.
        if not self.blocking or a1 is None:
            return False
        if a1.type() == QEvent.Type.Wheel:
            # Ignored, the wheel event propagates to the parent widgets.
            a1.ignore()
            return True
        if isinstance(a1, QKeyEvent) and a1.key() in _FOCUS_KEYS:
            return False
        return a1.type() in _BLOCKED_EVENTS


@dataclass(frozen=True)
class LocalText:
    """View input that state does not hold, shown in place of state's input.

    text is the unparsed or host-rejected typed text ("" for a control that
    has no text). error is its parse error or the host's rejection message.
    expression is whether Expr mode was selected. section is True when it was
    recorded on a model section's row (a rejected Enabled box, or a rejected
    option dropdown or command while the selected option draws fields),
    which carries only the error.
    """

    text: str
    error: str
    expression: bool
    section: bool


def _expression(field: FieldView) -> bool:
    return field.expression and isinstance(field.raw, dict) and "expr" in field.raw


def _rounded(field: FieldView, value: object) -> str:
    """str(value), with a decimals field's number rounded for display only."""
    if (
        field.decimals is not None
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ):
        return trimmed_decimals(float(value), field.decimals)
    return str(value)


def _input_text(field: FieldView) -> str:
    raw = field.raw
    if _expression(field) and isinstance(raw, dict):
        return str(raw.get("expr", ""))
    if not field.present and field.has_value and field.kind != "json":
        return "" if field.value is None else _rounded(field, field.value)
    if field.kind == "json":
        return json.dumps(raw) if field.present or raw is not None else ""
    if field.kind == "complex" and isinstance(raw, dict) and "__complex__" in raw:
        try:
            return str(decode_complex(raw))
        except ValueError:
            # Model-invalid envelopes remain visible rather than breaking build.
            return json.dumps(raw)
    return "" if raw is None else _rounded(field, raw)


def _shows_raw(field: FieldView, text: str) -> bool:
    """Whether text already reads as a decimals field's Direct number.

    That field shows rounded text; rewriting typed text with more digits
    would round them away under the cursor while the state keeps them.
    """
    raw = field.raw
    if (
        field.decimals is None
        or _expression(field)
        or not isinstance(raw, int | float)
        or isinstance(raw, bool)
    ):
        return False
    try:
        return float(text) == raw
    except ValueError:
        # Text that is not a number cannot already show the number.
        return False


def _parse(field: FieldView, text: str, *, expression: bool) -> JsonValue:
    """Parse view syntax only; all value constraints remain with FormState."""
    if expression:
        return {"expr": text}
    if not text.strip() and field.nullable:
        return None
    if field.kind == "integer":
        return int(text)
    if field.kind == "number":
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Enter a finite number")
        return value
    if field.kind == "complex":
        value = complex(text)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("Enter a finite complex number")
        return _JSON.validate_python(encode_complex(value))
    return _JSON.validate_json(text) if field.kind == "json" else text


def _direct_input(value: FormValue) -> JsonValue:
    """The Direct JSON input of a computed value, complex in its input codec.

    Raises ValueError for a value that has no JSON input, such as infinity.
    """
    if isinstance(value, complex):
        return _JSON.validate_python(encode_complex(value))
    return _JSON.validate_python(value)


def _mixed(base: QColor, accent: QColor, share: float) -> QColor:
    """base moved toward accent by share (0 keeps base, 1 gives accent)."""
    return QColor(
        round(base.red() + (accent.red() - base.red()) * share),
        round(base.green() + (accent.green() - base.green()) * share),
        round(base.blue() + (accent.blue() - base.blue()) * share),
    )


def _show_text_menu(widget: QLineEdit, action: QAction, pos: QPoint) -> None:
    """Open widget's standard context menu at pos with action appended."""
    menu = widget.createStandardContextMenu()
    if menu is None:
        raise RuntimeError("QLineEdit.createStandardContextMenu() returned None")
    menu.addSeparator()
    menu.addAction(action)
    menu.exec(widget.mapToGlobal(pos))
    menu.deleteLater()


def _missing_text(raw: JsonValue) -> str:
    """The dropdown text of an input no option matches, with MISSING_SUFFIX.

    A nonempty string shows as itself; other JSON (objects, numbers, null,
    "") shows as its JSON text.
    """
    text = raw if isinstance(raw, str) and raw else json.dumps(raw)
    return text + MISSING_SUFFIX


def _tight_label(text: str, parent: QWidget) -> QLabel:
    """A header label that never takes spare width, so controls sit beside it."""
    label = QLabel(text, parent)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
    return label


class _FoldTitle(QLabel):
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

    def __init__(
        self,
        label: str,
        body: QWidget,
        parent: QWidget,
        toggled: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._body = body
        self._toggled = toggled
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

    def _flip(self) -> None:
        self._body.setVisible(self._body.isHidden())
        self._show_mark()
        self._toggled()

    def _mark(self) -> str:
        return FOLDED_MARK if self._body.isHidden() else EXPANDED_MARK

    def _show_mark(self) -> None:
        self.setText(f"{self._mark()} {self._label}")


def _foldable_rows[K](
    parent: QWidget,
    folds: dict[K, bool],
    key: K,
    *,
    label: str,
    name: str,
    expanded: bool,
) -> tuple[QLabel, QGroupBox, QVBoxLayout]:
    """Build a fold title and the foldable box of rows it folds, under parent.

    Returns the title, named name and showing label after its fold mark;
    the box; and the box's rows layout, which has nested-group margins and
    ROW_SPACING. The caller puts the title in a header and the box below
    it. folds records user folds (True is expanded) across rebuilds: the
    box starts shown when folds[key] is True; without that key, when
    expanded is True. Each click on the title flips the box's visibility
    and the title's mark and records the new state as folds[key].
    """
    body = QGroupBox(parent)
    rows = QVBoxLayout(body)
    rows.setContentsMargins(*_GROUP_MARGINS)
    rows.setSpacing(ROW_SPACING)
    body.setVisible(folds.get(key, expanded))

    def record() -> None:
        folds[key] = not body.isHidden()

    title = _FoldTitle(label, body, parent, record)
    title.setObjectName(name)
    return title, body, rows


class FieldRow:
    """Keep one schema row stable while publication values change.

    field is the displayed FieldView; widget/layout own its Qt row. label is
    the left-aligned title at the start of the header; its parent sets one
    width for a model level's labels so their controls share a left edge.
    On a model row with rows to fold (every model row except a disabled
    nullable one) label is the bold fold title form.fold.<path>: a fold
    mark, then the field label, folding group on a click, or on Space,
    Return or Enter while it has Tab focus. Other rows,
    including a disabled nullable model, show the plain field label.
    input is its optional scalar control (read-only text for a fixed one-value
    Literal, which has no mode), or on an option row its one non-editable
    dropdown: each schema option's label with the option's value as item
    data, then, while a present input matches no option, the input's text
    plus MISSING_SUFFIX with the input as data. The dropdown's actions() are
    the selected option's commands, shown as its right-click menu. mode is
    the checkable Expression action of a text row that offers Expression
    input, checked in Expression mode; its text field lists it in actions()
    and its right-click menu, after the standard items. enabled toggles
    nullable models without options; group hosts model children, including
    the selected option's field rows, in its layout rows (both None for
    other rows, and for a disabled nullable model, which draws no box).
    The unit label comes right after the label and the row's input,
    dropdown or Enabled box, if any. value/error are public-named
    presentation labels, hidden while empty: value shows only an
    Expression-mode resolved value, error shows below the header. These controls never own or evaluate form state.
    Typed text is written when Enter is pressed or the field loses focus,
    unless it already shows state's input; switching mode writes at once,
    and switching a computed Expression to Direct writes its computed value.
    While display is called with editable=False the row is read-only: see
    display.
    """

    def __init__(
        self,
        field: FieldView,
        parent: QWidget,
        changed: Callable[[FormWritePath, Mapping[FormWritePath, JsonValue]], None],
        unparsed: Callable[[FormWritePath, LocalText], None],
        folds: dict[FormWritePath, bool],
        command: Callable[[FormWritePath, str], None],
    ) -> None:
        """Build under parent; call changed(path, batch), unparsed(path, text)
        or command(path, name).

        path is this row's field path. batch maps write paths to JSON input:
        one entry at path (an option row writes its chosen option's value).
        command receives the name of a triggered option command.
        Callbacks run synchronously for user input, never during display:
        for text, on Enter or focus loss while the row is editable, and not
        for text that already shows state's input.
        Child-model rows are built by the editor inside rows.
        folds is the editor's record of user folds by model path (True is
        expanded): a model row starts as folds says, else as the field's
        collapsed hint, and records each user fold there. Other rows ignore it.
        """
        self.field = field
        self._changed = changed
        self._unparsed = unparsed
        self._folds = folds
        self._command = command
        # The selected option's command actions on the dropdown, in order.
        self._commands: list[QAction] = []
        self._displaying = False
        # The view-local input this row shows instead of state's, if any.
        self._local: LocalText | None = None
        self.widget = QWidget(parent)
        # Installed on this row's dropdowns and checkboxes; display sets it.
        self._block = _UserInputBlock(self.widget)
        self.layout = QVBoxLayout(self.widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(_HEADER_SPACING)
        self.input: QWidget | None = None
        self.mode: QAction | None = None
        self.enabled: QCheckBox | None = None
        self.group: QGroupBox | None = None
        self.rows: QVBoxLayout | None = None
        header = QHBoxLayout()
        self.layout.addLayout(header)
        self.label = self._build_label(header)
        if field.options is not None:
            self._build_options(header)
        elif field.kind == "model":
            # A nullable option row lists None instead of an Enabled box.
            if field.nullable:
                self._build_enabled(header)
        elif field.fixed:
            self._build_fixed(header)
        else:
            self._build_input(header)
        if field.unit:
            header.addWidget(_tight_label(field.unit, self.widget))
        self.value = QLabel(self.widget)
        self.value.setObjectName(self._name("value"))
        self.value.hide()
        header.addWidget(self.value)
        if field.kind == "model" and field.options is None:
            # Keep the Enabled box and unit beside the title instead of
            # stretching them.
            header.addStretch()
        self.error = QLabel(self.widget)
        self.error.setObjectName(self._name("error"))
        self.error.setWordWrap(True)
        self.error.hide()
        self.layout.addWidget(self.error)
        hint = _MODE_HINT if self.mode is not None else ""
        self.widget.setToolTip("\n".join(filter(None, (field.description, hint))))

    def set_error(self, text: str) -> None:
        """Show text under the row's header; empty text hides the error label."""
        self.error.setText(text)
        self.error.setVisible(bool(text))

    def clear_value(self) -> None:
        """Clear and hide the resolved-value label, e.g. for unparsed local text."""
        self.value.clear()
        self.value.hide()

    def _name(self, kind: str) -> str:
        return f"form.{kind}." + ".".join(self.field.path)

    def _build_label(self, header: QHBoxLayout) -> QLabel:
        """Add the row's name label to header and return it.

        A model row with rows to fold gets the fold title instead, and its
        fold box goes into group and rows, below header.
        """
        field = self.field
        if field.kind != "model" or not field.active:
            label = _tight_label(field.label, self.widget)
            header.addWidget(label)
            return label
        title, self.group, self.rows = _foldable_rows(
            self.widget,
            self._folds,
            field.path,
            label=field.label,
            name=self._name("fold"),
            expanded=not field.collapsed,
        )
        header.addWidget(title)
        self.layout.addWidget(self.group)
        return title

    def _build_enabled(self, header: QHBoxLayout) -> None:
        self.enabled = QCheckBox("Enabled", self.widget)
        self.enabled.setObjectName(self._name("enabled"))
        self.enabled.installEventFilter(self._block)
        header.addWidget(self.enabled)
        self.enabled.toggled.connect(self._activate)

    def _build_options(self, header: QHBoxLayout) -> None:
        # Not editable: the GUI picks a listed option, while wire writes may
        # still hold any input, shown as a missing item.
        widget = QComboBox(self.widget)
        widget.setObjectName(self._name("input"))
        # Take the spare width like a line edit.
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        widget.installEventFilter(self._block)
        # The right-click menu lists the selected option's command actions.
        widget.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        widget.currentIndexChanged.connect(self._option_chosen)
        self.input = widget
        header.addWidget(widget)

    def _build_fixed(self, header: QHBoxLayout) -> None:
        # One legal value: nothing to choose or compute, so no mode and no
        # value signal. Without a frame or base color it reads as plain text,
        # not as an input box.
        widget = QLineEdit(self.widget)
        widget.setObjectName(self._name("input"))
        widget.setReadOnly(True)
        widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        widget.setFrame(False)
        widget.setStyleSheet("QLineEdit { background: transparent; }")
        self.input = widget
        header.addWidget(widget)

    def _option_chosen(self, index: int) -> None:
        options = self.field.options or ()
        # The missing item (after the options) writes nothing.
        if self._displaying or not 0 <= index < len(options):
            return
        # The core switches branches to fresh defaults; nothing is carried over.
        self._write(options[index].value)

    def _command_triggered(self, name: str) -> None:
        if not self._displaying:
            self._command(self.field.path, name)

    def _write(self, value: JsonValue) -> None:
        """Send value as this row's one-entry batch."""
        self._changed(self.field.path, {self.field.path: value})

    def _build_input(self, header: QHBoxLayout) -> None:
        # Numeric inputs use raw text, not spinbox ranges/validators: partial
        # text belongs to the view and model-invalid values must reach state.
        widget = make_value_widget(
            bool if self.field.kind == "boolean" else str, "", None
        )
        widget.setParent(self.widget)
        widget.setObjectName(self._name("input"))
        self.input = widget
        if isinstance(widget, QLineEdit):
            widget.setPlaceholderText(self.field.placeholder)
        else:
            widget.installEventFilter(self._block)
        header.addWidget(widget)
        if self.field.expression and isinstance(widget, QLineEdit):
            self.mode = self._build_mode(widget)
        # Text commits on Enter or focus loss, not per keystroke; a checkbox
        # commits each toggle.
        connect_committed_value_widget(widget, self._input_changed)

    def _build_mode(self, widget: QLineEdit) -> QAction:
        """Return the Expression action, listed on widget and after its menu."""
        mode = QAction("Expression", self.widget)
        mode.setObjectName(self._name("mode"))
        mode.setCheckable(True)
        mode.triggered.connect(self._mode_changed)
        widget.addAction(mode)
        # Keep the text field's standard menu; mode follows it.
        widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda pos: _show_text_menu(widget, mode, pos)
        )
        return mode

    def _activate(self) -> None:
        if not self._displaying and self.enabled is not None:
            self._write({} if self.enabled.isChecked() else None)

    def _mode_changed(self) -> None:
        if self._displaying:
            return
        expression = self._expression_selected()
        self._show_mode(expression=expression)
        if not expression and self._shows_computed_expression():
            try:
                direct = _direct_input(self.field.value)
            except ValueError:
                # No JSON input for this value: keep the text, as when the
                # expression does not compute.
                pass
            else:
                # Direct mode starts from the value the core computed.
                self._write(direct)
                return
        # The text stays; it is written again in the new mode.
        self._input_changed()

    def _shows_computed_expression(self) -> bool:
        """Whether the text field shows state's Expression input, computed.

        Only then is field.value the value of the shown text: uncommitted,
        unparseable or rejected text differs from state's input, and an
        expression that does not compute has no preview value.
        """
        widget = self.input
        return (
            isinstance(widget, QLineEdit)
            and self._local is None
            and _expression(self.field)
            and self.field.has_value
            and widget.text() == _input_text(self.field)
        )

    def _shows_state(self, text: str, *, expression: bool) -> bool:
        """Whether text in the given mode already shows state's input.

        Committing it would write the same input again. A decimals field's
        rounded text, or typed text equal to its number, also counts.
        """
        if self._local is not None or expression != _expression(self.field):
            return False
        return text == _input_text(self.field) or _shows_raw(self.field, text)

    def _expression_selected(self) -> bool:
        return self.mode is not None and self.mode.isChecked()

    def _show_mode(self, *, expression: bool) -> None:
        """Check mode and tint the current input for the given mode."""
        if self.mode is None or self.input is None:
            return
        self.mode.setChecked(expression)
        # An empty QPalette sets no role, so a Direct input inherits again.
        palette = QPalette()
        if expression:
            shown = self.widget.palette()
            palette.setColor(
                QPalette.ColorRole.Base,
                _mixed(
                    shown.color(QPalette.ColorRole.Base),
                    shown.color(QPalette.ColorRole.Highlight),
                    _EXPRESSION_TINT,
                ),
            )
        self.input.setPalette(palette)

    def _input_changed(self) -> None:
        if self._displaying:
            return
        widget = self.input
        if isinstance(widget, QCheckBox):
            self._write(widget.isChecked())
        elif isinstance(widget, QLineEdit):
            expression = self._expression_selected()
            text = widget.text()
            # Focus also leaves when the editor is detached (disabled) or
            # while the row is read-only; those commit nothing.
            if (
                widget.isReadOnly()
                or not widget.isEnabled()
                or self._shows_state(text, expression=expression)
            ):
                return
            try:
                parsed = _parse(self.field, text, expression=expression)
            except ValueError as exc:
                local = LocalText(text, str(exc), expression, section=False)
                self._local = local
                self._unparsed(self.field.path, local)
                return
            self._write(parsed)

    def rejected_input(self, message: str) -> LocalText:
        """Return the row's current input as local input the host rejected.

        message is the host's reason. Text inputs keep their typed text and
        Expr mode; other controls, including dropdowns, keep no text, so
        display shows state's value again. Reads widgets only; writes nothing.
        """
        widget = self.input
        text = widget.text() if isinstance(widget, QLineEdit) else ""
        expression = self._expression_selected()
        section = self.field.kind == "model"
        return LocalText(text, message, expression, section=section)

    def display(
        self, field: FieldView, local: LocalText | None, *, editable: bool
    ) -> None:
        """Show a same-shape publication, keeping local text and blocking writes.

        editable False makes the row read-only, not disabled: a text input is
        a read-only QLineEdit whose text can still be selected and copied;
        dropdowns and checkboxes (including Enabled) show their value in
        normal colours but ignore clicks, wheel turns and keys other than Tab;
        the Expression action is disabled. Folds still work. editable says
        nothing about model validity. local overrides
        the text and mode of a text input and hides its evaluated label;
        checkboxes and dropdowns still show state. The value label shows
        only a resolved Expression-mode value; Direct input is not echoed.
        Shape changes require a new row; metadata stays as built, and the
        fold stays as built or as the user last set it.
        """
        self.field = field
        self._local = local
        self._displaying = True
        try:
            self._display_input(local, editable=editable)
            shown = ""
            if local is None and field.has_value and _expression(field):
                shown = _rounded(field, field.value)
                if isinstance(field.value, str):
                    resolved = json.dumps(field.value, ensure_ascii=False)
                    shown = f"{_input_text(field)} = {resolved}"
            self.value.setText(shown)
            self.value.setVisible(bool(shown))
        finally:
            self._displaying = False

    def _display_options(
        self, options: tuple[OptionView, ...], *, editable: bool
    ) -> None:
        """Show options and select the input's; list its commands as actions.

        A present input that matches no option gets a selected missing item;
        an unfilled one selects nothing.
        """
        widget = self.input
        if not isinstance(widget, QComboBox):
            return
        field = self.field
        items: list[tuple[str, JsonValue]] = [
            (option.label, option.value) for option in options
        ]
        selected = -1 if field.selected is None else field.selected
        if field.selected is None and field.present:
            items.append((_missing_text(field.raw), field.raw))
            selected = len(items) - 1
        shown = [
            (widget.itemText(i), widget.itemData(i)) for i in range(widget.count())
        ]
        if shown != items:
            widget.clear()
            for item_text, item_data in items:
                widget.addItem(item_text, item_data)
        widget.setCurrentIndex(selected)
        listed = () if field.selected is None else options[field.selected].commands
        wanted = [
            (f"{self._name('command')}.{item.name}", item.label) for item in listed
        ]
        if [(a.objectName(), a.text()) for a in self._commands] != wanted:
            for action in self._commands:
                widget.removeAction(action)
                # Retire the name now; deleteLater waits until an active
                # trigger of this action finishes.
                action.setObjectName("")
                action.deleteLater()
            self._commands = []
            for item, (name, _) in zip(listed, wanted, strict=True):
                action = QAction(item.label, widget)
                action.setObjectName(name)
                action.triggered.connect(partial(self._command_triggered, item.name))
                widget.addAction(action)
                self._commands.append(action)
        for action in self._commands:
            action.setEnabled(editable)

    def _display_input(self, local: LocalText | None, *, editable: bool) -> None:
        field = self.field
        self._block.blocking = not editable
        if self.enabled is not None:
            self.enabled.setChecked(field.active)
        if field.options is not None:
            self._display_options(field.options, editable=editable)
            return
        expression = local.expression if local else _expression(field)
        if self.mode is not None:
            self.mode.setEnabled(editable)
        self._show_mode(expression=expression)
        widget = self.input
        if widget is None:
            return
        if isinstance(widget, QLineEdit):
            # A fixed one-value Literal stays read-only text either way.
            widget.setReadOnly(field.fixed or not editable)
            text = local.text if local else _input_text(field)
            # Avoid resetting cursor/selection on unchanged publications.
            if widget.text() != text and not _shows_raw(field, widget.text()):
                write_value_widget(widget, text)
        elif isinstance(widget, QCheckBox):
            write_value_widget(widget, field.raw)


class GroupBlock:
    """One foldable display group holding some rows of one model level.

    widget is the block, named form.group.<path>.<name> (form.group.<name>
    for the root level), where path is the model level's path and name the
    fields' Display group. Its header holds label, the bold fold title
    form.fold.group.<path>.<name> (form.fold.group.<name> at the root): a
    fold mark, then the group name, folding body on a click, or on Space,
    Return or Enter while it has Tab focus; the editor
    puts it in the level's label column. A GROUP_ERRORS_NOTE label follows
    it, hidden until set_has_errors(True). body is a QGroupBox, hidden while
    folded, whose layout rows takes the member rows; members are their paths
    in order. The block owns no form state.
    """

    def __init__(
        self,
        path: FormWritePath,
        name: str,
        members: tuple[FormWritePath, ...],
        parent: QWidget,
        folds: dict[tuple[FormWritePath, str], bool],
    ) -> None:
        """Build the block under parent; the caller adds widget to its layout.

        folds is the editor's record of user group folds by (path, name),
        True for expanded: the block starts as folds says, else folded, and
        records each user fold there. The caller adds member rows to rows.
        """
        self.members = members
        self.widget = QWidget(parent)
        self.widget.setObjectName("form.group." + ".".join((*path, name)))
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(_HEADER_SPACING)
        header = QHBoxLayout()
        layout.addLayout(header)
        self.label, self.body, self.rows = _foldable_rows(
            self.widget,
            folds,
            (path, name),
            label=name,
            name="form.fold.group." + ".".join((*path, name)),
            expanded=False,
        )
        header.addWidget(self.label)
        layout.addWidget(self.body)
        self._errors_note = QLabel(GROUP_ERRORS_NOTE, self.widget)
        self._errors_note.setStyleSheet(_ERRORS_NOTE_STYLE)
        self._errors_note.hide()
        header.addWidget(self._errors_note)
        # Keep the errors note beside the title instead of stretching it.
        header.addStretch()

    def set_has_errors(self, *, has_errors: bool) -> None:
        """Show the header's errors note when has_errors, else hide it.

        The fold does not change: a folded group stays folded.
        """
        self._errors_note.setVisible(has_errors)
