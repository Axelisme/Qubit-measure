"""Qt rows and JSON text parsing, independent of form evaluation/validation."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ConfigDict, JsonValue, TypeAdapter

# qtpy.QtGui.QAction resolves to NoReturn for pyright; PyQt6 is the binding.
from PyQt6.QtGui import QAction
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.cfg.codec import decode_complex, encode_complex
from zcu_tools.gui.form import FormWritePath
from zcu_tools.gui.widgets.cfg.fields.common import (
    connect_value_widget,
    make_value_widget,
    write_value_widget,
)
from zcu_tools.gui.widgets.spinbox import trimmed_decimals

from ._schema_view import FieldView

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
_ERRORS_NOTE_STYLE = "QLabel { color: #b3261e; }"


@dataclass(frozen=True)
class LocalText:
    """View input that state does not hold, shown in place of state's input.

    text is the unparsed or host-rejected typed text ("" for a control that
    has no text). error is its parse error or the host's rejection message.
    expression is whether Expr mode was selected. section is True when it was
    recorded on a model section's row (a rejected Enabled box or Key/Inline
    switch), which carries only the error.
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


def _tight_label(text: str, parent: QWidget) -> QLabel:
    """A header label that never takes spare width, so controls sit beside it."""
    label = QLabel(text, parent)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
    return label


def _fold_button[K](
    body: QWidget, parent: QWidget, folds: dict[K, bool], key: K, *, expanded: bool
) -> QPushButton:
    """Return an Expand/Collapse QPushButton under parent that folds body.

    folds records user folds (True is expanded) across rebuilt buttons. body
    starts shown when folds[key] is True; without that key, when expanded is
    True. Each click flips body's visibility and the button text and records
    the new state as folds[key].
    """
    shown = folds.get(key, expanded)
    body.setVisible(shown)
    button = QPushButton("Collapse" if shown else "Expand", parent)

    def clicked() -> None:
        now_expanded = body.isHidden()
        body.setVisible(now_expanded)
        button.setText("Collapse" if now_expanded else "Expand")
        folds[key] = now_expanded

    button.clicked.connect(clicked)
    return button


class FieldRow:
    """Keep one schema row stable while publication values change.

    field is the displayed FieldView; widget/layout own its Qt row. label is
    the left-aligned name label at the start of the header; its parent sets
    one width for a model level's labels so their controls share a left edge.
    input is its optional scalar control (read-only text for a fixed one-value
    Literal, which has no mode). mode is the checkable Expression action of a
    row that offers Expression input, checked in Expression mode; input lists
    it in actions() and its right-click menu, after a text field's standard
    items. A choice row's input is a dropdown of only the choices in Direct
    mode and a text field in Expression mode. switch selects Key or Inline
    on a str-or-model field; enabled toggles nullable models; group hosts
    model children. value/error are public-named presentation labels, hidden
    while empty: value shows only an Expression-mode resolved value, error
    shows below the header. These controls never own or evaluate form state.
    """

    def __init__(
        self,
        field: FieldView,
        parent: QWidget,
        changed: Callable[[FormWritePath, JsonValue], None],
        unparsed: Callable[[FormWritePath, LocalText], None],
        folds: dict[FormWritePath, bool],
    ) -> None:
        """Build under parent; call changed(path, JSON) or unparsed(path, text).

        Callbacks run synchronously for user input, never during display.
        Child-model rows are built by the editor inside group.layout().
        folds is the editor's record of user folds by model path (True is
        expanded): a model row starts as folds says, else as the field's
        collapsed hint, and records each user fold there. Other rows ignore it.
        """
        self.field = field
        self._changed = changed
        self._unparsed = unparsed
        self._folds = folds
        self._displaying = False
        # Choice shown when Expression mode returns to Direct (see _direct_choice).
        self._direct_raw: JsonValue = None
        self.widget = QWidget(parent)
        self.layout = QVBoxLayout(self.widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(_HEADER_SPACING)
        self.input: QWidget | None = None
        self.mode: QAction | None = None
        self.switch: QComboBox | None = None
        self.enabled: QCheckBox | None = None
        self.group: QGroupBox | None = None
        header = QHBoxLayout()
        self.layout.addLayout(header)
        self.label = _tight_label(field.label, self.widget)
        header.addWidget(self.label)
        if field.key_or_inline:
            self._build_switch(header)
        if field.kind == "model":
            self._build_group(header)
        elif field.kind == "key":
            self._build_key(header)
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
        if field.kind == "model":
            # Keep the fold button beside its label instead of stretching it.
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

    def _build_group(self, header: QHBoxLayout) -> None:
        if self.field.nullable:
            self.enabled = QCheckBox("Enabled", self.widget)
            self.enabled.setObjectName(self._name("enabled"))
            header.addWidget(self.enabled)
            self.enabled.toggled.connect(self._activate)
        self.group = QGroupBox(self.widget)
        rows = QVBoxLayout(self.group)
        rows.setContentsMargins(*_GROUP_MARGINS)
        rows.setSpacing(ROW_SPACING)
        fold = _fold_button(
            self.group,
            self.widget,
            self._folds,
            self.field.path,
            expanded=not self.field.collapsed,
        )
        header.addWidget(fold)
        self.layout.addWidget(self.group)

    def _build_switch(self, header: QHBoxLayout) -> None:
        # Named form.mode.<path> like the Expression action of value rows,
        # but a QComboBox with data key/inline; such a row has no Expression.
        self.switch = QComboBox(self.widget)
        self.switch.setObjectName(self._name("mode"))
        self.switch.addItem("Key", "key")
        self.switch.addItem("Inline", "inline")
        self.switch.setCurrentIndex(1 if self.field.kind == "model" else 0)
        self.switch.currentIndexChanged.connect(self._switch_changed)
        header.addWidget(self.switch)

    def _build_key(self, header: QHBoxLayout) -> None:
        # A library key is free text; captured keys are suggestions in an
        # editable combo, so a key outside them stays typeable.
        widget = make_value_widget(str, "", self.field.options)
        widget.setParent(self.widget)
        widget.setObjectName(self._name("input"))
        if isinstance(widget, QComboBox):
            widget.setEditable(True)
            # Take the spare width like a line edit, leaving the switch compact.
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            widget.editTextChanged.connect(self._key_changed)
        elif isinstance(widget, QLineEdit):
            widget.setPlaceholderText(self.field.placeholder)
            widget.textChanged.connect(self._key_changed)
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

    def _switch_changed(self, _index: int) -> None:
        if self._displaying or self.switch is None:
            return
        # Each mode starts from its own default; the other mode's value is
        # never converted: "" is an empty key, {} the inline model defaults.
        inline = self.switch.currentData() == "inline"
        self._changed(self.field.path, {} if inline else "")

    def _key_changed(self, text: str) -> None:
        if not self._displaying:
            self._changed(self.field.path, text)

    def _build_input(self, header: QHBoxLayout) -> None:
        # Numeric inputs use raw text, not spinbox ranges/validators: partial
        # text belongs to the view and model-invalid values must reach state.
        widget = make_value_widget(
            bool if self.field.kind == "boolean" else str,
            "",
            self.field.options,
        )
        widget.setParent(self.widget)
        widget.setObjectName(self._name("input"))
        self.input = widget
        if isinstance(widget, QLineEdit):
            widget.setPlaceholderText(self.field.placeholder)
        if self.field.expression:
            # Owned by the row, so it outlives a choice row's input swaps.
            self.mode = QAction("Expression", self.widget)
            self.mode.setObjectName(self._name("mode"))
            self.mode.setCheckable(True)
            self.mode.triggered.connect(self._mode_changed)
        self._input_layout = header
        header.addWidget(widget)
        self._offer_mode(widget)
        connect_value_widget(widget, self._input_changed)

    def _offer_mode(self, widget: QWidget) -> None:
        """List mode on widget and in its right-click menu."""
        mode = self.mode
        if mode is None:
            return
        widget.addAction(mode)
        if isinstance(widget, QLineEdit):
            # Keep the text field's standard menu; mode follows it.
            widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            widget.customContextMenuRequested.connect(
                lambda pos: _show_text_menu(widget, mode, pos)
            )
        else:
            # A dropdown has no standard menu: its menu is its actions.
            widget.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

    def _activate(self) -> None:
        if not self._displaying and self.enabled is not None:
            self._changed(self.field.path, {} if self.enabled.isChecked() else None)

    def _mode_changed(self) -> None:
        if self._displaying:
            return
        expression = self._expression_selected()
        self._displaying = True
        try:
            if self.field.options is not None:
                # Expression text starts from the direct choice; no choice is "".
                text = (
                    self.input.text()
                    if isinstance(self.input, QLineEdit)
                    else ("" if self.field.raw is None else str(self.field.raw))
                )
                self._select_input(expression=expression)
                if isinstance(self.input, QLineEdit):
                    self.input.setText(text)
                else:
                    self._display_choices(self._direct_choice())
            self._show_mode(expression=expression)
        finally:
            self._displaying = False
        self._input_changed()

    def _expression_selected(self) -> bool:
        return self.mode is not None and self.mode.isChecked()

    def _direct_choice(self) -> object:
        """Choice to select when Expression mode returns to Direct.

        A resolved expression value that is a current option wins. Otherwise
        the last Direct choice returns, because an expression made from the
        direct text (for example "mA") need not resolve.
        """
        options = self.field.options or ()
        if self.field.has_value and self.field.value in options:
            return self.field.value
        return self._direct_raw

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

    def _select_input(self, *, expression: bool) -> None:
        if self.field.options is None:
            return
        use_text = expression
        if use_text == isinstance(self.input, QLineEdit):
            return
        old = self.input
        widget = make_value_widget(str, "", None if use_text else self.field.options)
        widget.setParent(self.widget)
        widget.setObjectName(self._name("input"))
        if isinstance(widget, QLineEdit):
            widget.setPlaceholderText(self.field.placeholder)
        if old is not None:
            self._input_layout.replaceWidget(old, widget)
            old.setParent(None)
            old.deleteLater()
        self.input = widget
        self._offer_mode(widget)
        connect_value_widget(widget, self._input_changed)

    def _display_choices(self, selected: object) -> None:
        widget = self.input
        if isinstance(widget, QComboBox):
            widget.clear()
            for value in self.field.options or ():
                widget.addItem(str(value), value)
            widget.setCurrentIndex(
                next(
                    (
                        index
                        for index, value in enumerate(self.field.options or ())
                        if value == selected
                    ),
                    -1,
                )
            )

    def _input_changed(self) -> None:
        if self._displaying:
            return
        widget = self.input
        if isinstance(widget, QCheckBox):
            self._changed(self.field.path, widget.isChecked())
        elif isinstance(widget, QComboBox):
            index = widget.currentIndex()
            options = self.field.options or ()
            if 0 <= index < len(options):
                self._changed(self.field.path, options[index])
        elif isinstance(widget, QLineEdit):
            expression = self._expression_selected()
            text = widget.text()
            try:
                parsed = _parse(self.field, text, expression=expression)
            except ValueError as exc:
                local = LocalText(text, str(exc), expression, section=False)
                self._unparsed(self.field.path, local)
                return
            self._changed(self.field.path, parsed)

    def rejected_input(self, message: str) -> LocalText:
        """Return the row's current input as local input the host rejected.

        message is the host's reason. Text and key inputs keep their typed
        text and Expr mode; other controls keep no text, so display shows
        state's value again. Reads widgets only; writes nothing.
        """
        widget = self.input
        if isinstance(widget, QLineEdit):
            text = widget.text()
        elif isinstance(widget, QComboBox) and widget.isEditable():
            text = widget.currentText()
        else:
            text = ""
        expression = self._expression_selected()
        section = self.field.kind == "model"
        return LocalText(text, message, expression, section=section)

    def display(
        self, field: FieldView, local: LocalText | None, *, editable: bool
    ) -> None:
        """Show a same-shape publication, keeping local text and blocking writes.

        editable controls input enablement, not model validity. local overrides
        the text and mode of a text or key input and hides its evaluated label;
        checkboxes, dropdowns and switches still show state. The value label shows
        only a resolved Expression-mode value; Direct input is not echoed.
        Shape changes require a new row; metadata stays as built, and the
        fold stays as built or as the user last set it.
        """
        self.field = field
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

    def _display_key(self, local: LocalText | None, *, editable: bool) -> None:
        widget = self.input
        if local is not None:
            text = local.text
        else:
            text = "" if self.field.raw is None else str(self.field.raw)
        if isinstance(widget, QComboBox):
            widget.setEnabled(editable)
            keys = [str(key) for key in self.field.options or ()]
            if [widget.itemText(index) for index in range(widget.count())] != keys:
                widget.clear()
                widget.addItems(keys)
            # Avoid resetting cursor/selection on unchanged publications.
            if widget.currentText() != text:
                widget.setEditText(text)
        elif isinstance(widget, QLineEdit):
            widget.setEnabled(editable)
            if widget.text() != text:
                widget.setText(text)

    def _display_input(self, local: LocalText | None, *, editable: bool) -> None:
        field = self.field
        if self.switch is not None:
            self.switch.setCurrentIndex(1 if field.kind == "model" else 0)
            self.switch.setEnabled(editable)
        if field.kind == "key":
            self._display_key(local, editable=editable)
            return
        expression = local.expression if local else _expression(field)
        if field.options is not None and not expression:
            self._direct_raw = field.raw
        if self.enabled is not None:
            self.enabled.setChecked(field.active)
            self.enabled.setEnabled(editable)
        if self.mode is not None:
            self.mode.setEnabled(editable)
        self._select_input(expression=expression)
        self._show_mode(expression=expression)
        widget = self.input
        if widget is None:
            return
        widget.setEnabled(editable)
        if isinstance(widget, QComboBox):
            self._display_choices(field.raw)
        elif isinstance(widget, QLineEdit):
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
    fields' Display group. Its header holds label, the group name, which the
    editor puts in the level's label column; the Expand/Collapse fold button;
    and a GROUP_ERRORS_NOTE label, hidden until set_has_errors(True). body is
    a QGroupBox, hidden while folded, whose layout rows takes the member
    rows; members are their paths in order. The block owns no form state.
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
        self.label = _tight_label(name, self.widget)
        header.addWidget(self.label)
        self.body = QGroupBox(self.widget)
        self.rows = QVBoxLayout(self.body)
        self.rows.setContentsMargins(*_GROUP_MARGINS)
        self.rows.setSpacing(ROW_SPACING)
        header.addWidget(
            _fold_button(self.body, self.widget, folds, (path, name), expanded=False)
        )
        self._errors_note = QLabel(GROUP_ERRORS_NOTE, self.widget)
        self._errors_note.setStyleSheet(_ERRORS_NOTE_STYLE)
        self._errors_note.hide()
        header.addWidget(self._errors_note)
        # Keep the fold button beside its label instead of stretching it.
        header.addStretch()
        layout.addWidget(self.body)

    def set_has_errors(self, *, has_errors: bool) -> None:
        """Show the header's errors note when has_errors, else hide it.

        The fold does not change: a folded group stays folded.
        """
        self._errors_note.setVisible(has_errors)
