"""Qt rows and JSON text parsing, independent of form evaluation/validation."""

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial

from pydantic import ConfigDict, JsonValue, TypeAdapter

# qtpy.QtGui.QAction resolves to NoReturn for pyright; PyQt6 is the binding.
from PyQt6.QtGui import QAction
from qtpy.QtCore import QEvent, QObject, QPoint, Qt
from qtpy.QtGui import (
    QKeyEvent,
    QPalette,
)
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.core.complex_codec import decode_complex, encode_complex
from zcu_tools.gui.form import FormOptionDescription, FormValue, FormWritePath
from zcu_tools.gui.widgets import ERROR_RED_LIGHT, error_colour, mixed_colour
from zcu_tools.gui.widgets.form.value_widgets import (
    FIELD_INPUT_MIN_WIDTH,
    connect_committed_value_widget,
    make_value_widget,
    write_value_widget,
)
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.gui.widgets.spinbox import trimmed_decimals

from ._fold import FoldTitle, foldable_rows
from ._text_input import TextInput as _TextInput
from ._view import FieldView

_JSON = TypeAdapter(JsonValue, config=ConfigDict(allow_inf_nan=False))
# Pixels between a row's header and its nested group.
_HEADER_SPACING = 2
# Tooltip line on rows whose input has the right-click Expression action.
_MODE_HINT = "Right-click to switch Direct / Expression"
# Share of the palette's Highlight mixed into an Expression input's Base colour:
# a light tint of the theme's own colours, so text stays readable in light
# and dark palettes alike.
_EXPRESSION_TINT = 0.18
# Fixed red mixed into Base for Direct input errors, independent of Highlight.
_ERROR_TINT = 0.18
# Shown in a group block's header while any of its rows shows an error, so a
# folded group still tells the user where Start is blocked.
GROUP_ERRORS_NOTE = "Contains errors"
# Suffix of the option dropdown item that shows an input no option matches.
MISSING_SUFFIX = " (missing)"
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
    """Whether text already reads as the state's Direct number.

    Typed "11" commits as 11.0 and a decimals row shows rounded text; rewriting
    the text under the cursor would turn "11" into "11.0" mid-typing or round
    digits away while the state keeps them.
    """
    raw = field.raw
    if _expression(field) or not isinstance(raw, int | float) or isinstance(raw, bool):
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


def _show_text_menu(widget: QLineEdit, actions: list[QAction], pos: QPoint) -> None:
    """Open widget's standard context menu at pos with action appended."""
    menu = widget.createStandardContextMenu()
    if menu is None:
        raise RuntimeError("QLineEdit.createStandardContextMenu() returned None")
    menu.addSeparator()
    menu.addActions(actions)
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


@dataclass(frozen=True)
class RowActions:
    """Explicit row operations, separate from parsed value input."""

    command: Callable[[FormWritePath, str], None]
    select: Callable[[FormWritePath, JsonValue], None]


class FieldRow:
    """Keep one described row stable while publication values change.

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
    dropdown: each described option's label with the option's value as item
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
    dropdown or Enabled box, if any. value is a public-named label, hidden
    while empty, showing only an Expression-mode resolved value. Errors tint
    the input or name label and lead the row tooltip, without adding height.
    The coloured widget carries the boolean formError property.
    These controls never own or evaluate form state.
    Typed text is written after 500 ms idle, on Enter or ordinary focus loss,
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
        actions: RowActions,
    ) -> None:
        """Build under parent with synchronous edit callbacks.

        changed(path, batch) writes parsed input; batch contains the row path.
        unparsed(path, text) reports view-local drafts. select(path, value) sends
        the declared option value, separately from raw value writes.
        command(path, name) sends a triggered option command.
        Callbacks run synchronously for user input, never during display:
        for text, on Enter or focus loss while the row is editable, and not
        for text that already shows state's input.
        Child-model rows are built by the editor inside rows.
        folds is the editor's record of user folds by model path (True is
        expanded): a model row starts as folds says, else as the field's
        collapsed hint, and records each user fold there. Other rows ignore it.
        """
        self.field = field
        self._set_idle_commit: Callable[[bool], None] = lambda _enabled: None
        self._changed = changed
        self._unparsed = unparsed
        self._folds = folds
        self._command = actions.command
        self._select = actions.select
        # The selected option's command actions on the dropdown, in order.
        self._commands: list[QAction] = []
        self._field_actions: list[QAction] = []
        self._displaying = False
        self._fields_only = False
        self._picked: Callable[[], None] = lambda: None
        # The view-local input this row shows instead of state's, if any.
        self._local: LocalText | None = None
        self._error = ""
        self.widget = QWidget(parent)
        self.widget.setObjectName(self._name("row"))
        # Installed on this row's dropdowns and checkboxes; display sets it.
        self._block = _UserInputBlock(self.widget)
        self.layout = QVBoxLayout(self.widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(_HEADER_SPACING)
        self.input: QWidget | None = None
        self.picker: QComboBox | None = None
        self.mode: QAction | None = None
        self.enabled: QCheckBox | None = None
        self.group: QGroupBox | None = None
        self.rows: QVBoxLayout | None = None
        header = QHBoxLayout()
        self.layout.addLayout(header)
        self.label = self._build_label(header)
        self._sync_field_commands(editable=True)
        self._build_control(header)
        if field.unit:
            header.addWidget(_tight_label(field.unit, self.widget))
        self.value = QLabel(self.widget)
        make_label_selectable(self.value)
        self.value.setObjectName(self._name("value"))
        self.value.hide()
        header.addWidget(self.value)
        if field.kind in ("model", "elements") and field.options is None:
            # Keep the Enabled box and unit beside the title instead of
            # stretching them.
            header.addStretch()
        self._input_style = self.input.styleSheet() if self.input is not None else ""
        self._input_minimum_height = (
            self.input.minimumHeight() if self.input is not None else 0
        )
        self._input_native_height = (
            self.input.sizeHint().height() if self.input is not None else 0
        )
        hint = _MODE_HINT if self.mode is not None else ""
        self._tooltip = "\n".join(filter(None, (field.description, hint)))
        self.set_error("")

    def set_idle_commit(self, *, enabled: bool) -> None:
        """Enable or cancel text idle commits without changing pending text."""
        self._set_idle_commit(enabled)

    def set_completion_candidates(self, candidates: tuple[str, ...]) -> None:
        """Pass captured $alias and one-level field names to this row's text input.

        candidates is sorted, case-sensitive and detached from source services.
        Non-text rows ignore it; Direct and read-only text never offer hints.
        """
        if isinstance(self.input, _TextInput):
            self.input.set_completion_candidates(candidates)

    def connect_fold(self, callback: Callable[[], None]) -> None:
        """Call callback after a user fold, if this row has a fold title.

        The callback takes no arguments, runs on the GUI thread and may build
        children on first expansion. Connecting never calls it immediately.
        A leaf or disabled model has no fold signal and does nothing.
        """
        if isinstance(self.label, FoldTitle):
            self.label.fold_changed.connect(callback)

    def connect_pick(self, callback: Callable[[], None]) -> None:
        """Call callback after the user picks another element, if this is a picker row.

        The callback takes no arguments and runs on the GUI thread; connecting
        never calls it. Display refreshes never call it.
        """
        self._picked = callback

    def picked(self) -> int:
        """Return the picked element index, or -1 on a row without a picker."""
        return -1 if self.picker is None else self.picker.currentIndex()

    def configure_body(self, *, element: bool) -> None:
        """Configure a freshly built row as an ordinary row or picked element.

        element=False keeps the ordinary header and fold. element=True omits
        the title and extra indentation, keeping the body expanded independently
        of model folds. Its fields keep their labels, paths and fold controls.
        """
        if not element:
            return
        self._fields_only = True
        self.label.hide()
        if self.group is not None and self.rows is not None:
            self.group.show()
            self.rows.setContentsMargins(0, 0, 0, 0)

    def set_children_available(self, *, available: bool) -> None:
        """Hide an empty model body's fold marker, or restore it for children.

        available means the first expanded description has drawn children.
        The body and title remain owned by this row for later publications;
        empty bodies stay hidden, and no write or fold notification occurs.
        The public fold object name exists only when the layer can be folded.
        """
        if self._fields_only:
            if self.group is not None:
                self.group.setVisible(available)
            return
        if isinstance(self.label, FoldTitle):
            self.label.setObjectName(self._name("fold") if available else "")
            self.label.set_foldable(
                available=available,
                expanded=self._folds.get(self.field.path, not self.field.collapsed),
            )

    def set_error(self, text: str) -> None:
        """Mark the input or name label; lead the row tooltip with text.

        Empty text restores the normal mode colours and description/hint.
        No error label or extra layout space is added.
        """
        self._error = text
        self.widget.setToolTip("\n\n".join(filter(None, (text, self._tooltip))))
        self._restyle_input()

    def _restyle_input(self) -> None:
        """Apply the current mode tint and error styling without stacking colours."""
        invalid = bool(self._error)
        widget = self.input
        if not isinstance(widget, QLineEdit | QComboBox):
            self.label.setProperty("formError", invalid)
            palette = QPalette()
            if invalid:
                palette.setColor(
                    QPalette.ColorRole.WindowText, error_colour(self.widget)
                )
            self.label.setPalette(palette)
            return
        widget.setProperty("formError", invalid)
        expression = self._expression_selected()
        # Always mix the inherited Base, not an already tinted input palette.
        shown = self.widget.palette()
        palette = QPalette()
        if expression or invalid:
            accent = (
                shown.color(QPalette.ColorRole.Highlight)
                if expression
                else ERROR_RED_LIGHT
            )
            share = _EXPRESSION_TINT if expression else _ERROR_TINT
            background = mixed_colour(
                shown.color(QPalette.ColorRole.Base), accent, share
            )
            palette.setColor(QPalette.ColorRole.Base, background)
            if isinstance(widget, QComboBox):
                # Non-editable dropdowns paint Button rather than Base.
                palette.setColor(QPalette.ColorRole.Button, background)
        # A fixed Literal normally uses a transparent background, but its
        # invalid input must still show the error's Base tint.
        style = "" if invalid and self.field.fixed else self._input_style
        bordered = expression and invalid
        if bordered:
            # A border-only Qt stylesheet paints its own default background,
            # not the input's Base. Keep the Expression interior explicit.
            background = palette.color(QPalette.ColorRole.Base).name()
            style += (
                f"QLineEdit {{ border: 1px solid {error_colour(self.widget).name()}; "
                f"background-color: {background}; }}"
            )
        widget.setStyleSheet(style)
        # A stylesheet border has a smaller size hint than the native frame.
        # Retain the native height so errors never move neighbouring rows.
        widget.setMinimumHeight(
            max(self._input_minimum_height, self._input_native_height)
            if bordered
            else self._input_minimum_height
        )
        widget.setPalette(palette)

    def clear_value(self) -> None:
        """Clear and hide the resolved-value label, e.g. for unparsed local text."""
        self.value.clear()
        self.value.hide()

    def _name(self, kind: str) -> str:
        return f"form.{kind}." + ".".join(str(part) for part in self.field.path)

    def _build_label(self, header: QHBoxLayout) -> QLabel:
        """Add the row's name label to header and return it.

        A model row with rows to fold gets the fold title instead, and its
        fold box goes into group and rows, below header.
        """
        field = self.field
        if not field.can_expand:
            label = _tight_label(field.label, self.widget)
            header.addWidget(label)
            return label
        title, self.group, self.rows = foldable_rows(
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

    def _sync_field_commands(self, *, editable: bool) -> None:
        """Keep the mounted row's context actions aligned with its description."""
        listed = self.field.commands
        if [(a.objectName(), a.text()) for a in self._field_actions] != [
            (f"{self._name('command')}.{item.name}", item.label) for item in listed
        ]:
            for action in self._field_actions:
                self.label.removeAction(action)
                action.setObjectName("")
                action.deleteLater()
            self._field_actions = []
            for item in listed:
                action = QAction(item.label, self.label)
                action.setObjectName(f"{self._name('command')}.{item.name}")
                action.triggered.connect(partial(self._command_triggered, item.name))
                self.label.addAction(action)
                self._field_actions.append(action)
        if listed:
            self.label.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        for action in self._field_actions:
            action.setEnabled(editable)

    def _build_control(self, header: QHBoxLayout) -> None:
        """Build the input or navigation control selected by this row's metadata."""
        field = self.field
        if field.options is not None:
            self._build_options(header)
        elif field.kind == "elements":
            self._build_picker(header)
        elif field.kind == "model":
            # A nullable option row lists None instead of an Enabled box.
            if field.nullable:
                self._build_enabled(header)
        elif field.fixed:
            self._build_fixed(header)
        else:
            self._build_input(header)

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

    def _build_picker(self, header: QHBoxLayout) -> None:
        # Navigation only: choosing an element writes nothing and never reads
        # the picked element's text, so it stays usable on a read-only row.
        widget = QComboBox(self.widget)
        widget.setObjectName(self._name("picker"))
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        widget.currentIndexChanged.connect(self._pick_changed)
        self.picker = widget
        header.addWidget(widget)

    def _pick_changed(self) -> None:
        if not self._displaying:
            self._picked()

    def _display_picker(self) -> None:
        """Show each element as "index: name", keeping the user's pick."""
        widget = self.picker
        if widget is None:
            return
        labels = [f"{item.index}: {item.name}" for item in self.field.elements or ()]
        if [widget.itemText(i) for i in range(widget.count())] != labels:
            previous = max(widget.currentIndex(), 0)
            widget.clear()
            widget.addItems(labels)
            widget.setCurrentIndex(min(previous, len(labels) - 1))

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
        self._select(self.field.path, options[index].value)

    def _command_triggered(self, name: str) -> None:
        if not self._displaying:
            self._command(self.field.path, name)

    def _write(self, value: JsonValue) -> None:
        """Send value as this row's one-entry batch."""
        self._changed(self.field.path, {self.field.path: value})

    def _build_input(self, header: QHBoxLayout) -> None:
        # Numeric inputs use raw text, not spinbox ranges/validators: partial
        # text belongs to the view and model-invalid values must reach state.
        if self.field.kind == "boolean":
            widget = make_value_widget(bool, "", None)
            widget.setParent(self.widget)
        else:
            widget = _TextInput(self.widget)
            widget.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        widget.setObjectName(self._name("input"))
        self.input = widget
        if isinstance(widget, QLineEdit):
            widget.setPlaceholderText(self.field.placeholder)
        else:
            widget.installEventFilter(self._block)
        header.addWidget(widget)
        if self.field.expression and isinstance(widget, QLineEdit):
            self.mode = self._build_mode(widget)
        # Shared text policy commits on idle/Enter/focus loss; a checkbox
        # commits each toggle.
        self._set_idle_commit = connect_committed_value_widget(
            widget, self._input_changed
        )

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
            lambda pos: _show_text_menu(widget, [mode, *self._commands], pos)
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

    def has_pending_text(self) -> bool:
        """Return whether the text control differs from published state input."""
        return isinstance(self.input, QLineEdit) and not self._shows_state(
            self.input.text(), expression=self._expression_selected()
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
        if isinstance(self.input, _TextInput):
            self.input.set_expression_mode(expression=expression)
        self._restyle_input()

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
        resolved Expressions, or an Effective value when numeric preview differs
        from authored Direct input. Equal Direct values are not echoed.
        Shape changes require a new row; metadata stays as built, and the
        fold stays as built or as the user last set it.
        """
        self.field = field
        self._sync_field_commands(editable=editable)
        for action in self._commands:
            action.setEnabled(editable)
        self._local = local
        self._displaying = True
        try:
            self._display_input(local, editable=editable)
            self._display_picker()
            shown = ""
            effective = (
                field.kind in ("integer", "number", "complex")
                and field.present
                and not _expression(field)
                and field.value != field.raw
            )
            if local is None and field.has_value and (_expression(field) or effective):
                resolved = (
                    json.dumps(field.value, ensure_ascii=False)
                    if isinstance(field.value, str)
                    else _rounded(field, field.value)
                )
                shown = f"Effective: {resolved}" if effective else f"= {resolved}"
            self.value.setText(shown)
            self.value.setVisible(bool(shown))
        finally:
            self._displaying = False

    def _display_options(
        self, options: tuple[FormOptionDescription, ...], *, editable: bool
    ) -> None:
        """Show options and select the input's; list its commands as actions.

        A present selection that matches no option gets a selected missing
        item; an unfilled one selects nothing. Raw child input is independent.
        """
        widget = self.input
        if not isinstance(widget, QComboBox):
            return
        field = self.field
        items: list[tuple[str, JsonValue]] = [
            (option.label, option.value) for option in options
        ]
        selected = -1 if field.selected is None else field.selected
        if field.selected is None and field.selection_present:
            items.append((_missing_text(field.selection), field.selection))
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
            if isinstance(widget, _TextInput):
                widget.set_expression_mode(expression=expression)
            text = local.text if local else _input_text(field)
            # Avoid resetting cursor/selection on unchanged publications. A
            # local draft always shows; only state redraws keep equal numbers.
            if widget.text() != text and (
                local is not None or not _shows_raw(field, widget.text())
            ):
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
        toggled: Callable[[], None],
    ) -> None:
        """Build the block under parent; the caller adds widget to its layout.

        folds is the editor's record of user group folds by (path, name),
        True for expanded: the block starts as folds says, else folded, and
        records each user fold there. toggled runs after that change so the
        caller can lazily add member rows to rows on first expansion.
        """
        self.members = members
        self.widget = QWidget(parent)
        self.widget.setObjectName(
            "form.group." + ".".join(str(part) for part in (*path, name))
        )
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(_HEADER_SPACING)
        header = QHBoxLayout()
        layout.addLayout(header)
        self.label, self.body, self.rows = foldable_rows(
            self.widget,
            folds,
            (path, name),
            label=name,
            name="form.fold.group." + ".".join(str(part) for part in (*path, name)),
            expanded=False,
        )
        self.label.fold_changed.connect(toggled)
        header.addWidget(self.label)
        layout.addWidget(self.body)
        self._errors_note = QLabel(GROUP_ERRORS_NOTE, self.widget)
        self._errors_note.hide()
        header.addWidget(self._errors_note)
        # Keep the errors note beside the title instead of stretching it.
        header.addStretch()

    def set_has_errors(self, *, has_errors: bool) -> None:
        """Show the header's errors note when has_errors, else hide it.

        The fold does not change: a folded group stays folded.
        """
        colour = error_colour(self.widget)
        self._errors_note.setStyleSheet(f"QLabel {{ color: {colour.name()}; }}")
        self._errors_note.setVisible(has_errors)
