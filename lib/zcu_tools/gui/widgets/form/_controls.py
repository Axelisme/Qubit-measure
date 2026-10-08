"""Qt rows and JSON text parsing, independent of form evaluation/validation."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ConfigDict, JsonValue, TypeAdapter
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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

from ._schema_view import FieldView

_JSON = TypeAdapter(JsonValue, config=ConfigDict(allow_inf_nan=False))


@dataclass(frozen=True)
class LocalText:
    """Unparsed view text, its syntax error, and the selected Expr mode."""

    text: str
    error: str
    expression: bool


def _expression(field: FieldView) -> bool:
    return field.expression and isinstance(field.raw, dict) and "expr" in field.raw


def _input_text(field: FieldView) -> str:
    raw = field.raw
    if _expression(field) and isinstance(raw, dict):
        return str(raw.get("expr", ""))
    if not field.present and field.has_value and field.kind != "json":
        return "" if field.value is None else str(field.value)
    if field.kind == "json":
        return json.dumps(raw) if field.present or raw is not None else ""
    if field.kind == "complex" and isinstance(raw, dict) and "__complex__" in raw:
        try:
            return str(decode_complex(raw))
        except ValueError:
            # Model-invalid envelopes remain visible rather than breaking build.
            return json.dumps(raw)
    return "" if raw is None else str(raw)


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


class FieldRow:
    """Keep one schema row stable while publication values change.

    field is the displayed FieldView; widget/layout own its Qt row. input is
    its optional scalar control; mode selects direct/Expr; enabled toggles
    nullable models; group hosts model children. value/error are public-named
    presentation labels. These controls never own or evaluate form state.
    """

    def __init__(
        self,
        field: FieldView,
        parent: QWidget,
        changed: Callable[[FormWritePath, JsonValue], None],
        unparsed: Callable[[FormWritePath, LocalText], None],
    ) -> None:
        """Build under parent; call changed(path, JSON) or unparsed(path, text).

        Callbacks run synchronously for user input, never during display.
        Child-model rows are built by the editor inside group.layout().
        """
        self.field = field
        self._changed = changed
        self._unparsed = unparsed
        self._displaying = False
        self.widget = QWidget(parent)
        self.layout = QVBoxLayout(self.widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.input: QWidget | None = None
        self.mode: QComboBox | None = None
        self.enabled: QCheckBox | None = None
        self.group: QGroupBox | None = None
        header = QHBoxLayout()
        self.layout.addLayout(header)
        header.addWidget(QLabel(field.label, self.widget))
        if field.kind == "model":
            self._build_group(header)
        else:
            self._build_input(header)
        if field.unit:
            header.addWidget(QLabel(field.unit, self.widget))
        self.value = QLabel(self.widget)
        self.value.setObjectName(self._name("value"))
        header.addWidget(self.value)
        self.error = QLabel(self.widget)
        self.error.setObjectName(self._name("error"))
        self.error.setWordWrap(True)
        self.layout.addWidget(self.error)
        self.widget.setToolTip(field.description)

    def _name(self, kind: str) -> str:
        return f"form.{kind}." + ".".join(self.field.path)

    def _build_group(self, header: QHBoxLayout) -> None:
        if self.field.nullable:
            self.enabled = QCheckBox("Enabled", self.widget)
            self.enabled.setObjectName(self._name("enabled"))
            header.addWidget(self.enabled)
            self.enabled.toggled.connect(self._activate)
        self.group = QGroupBox(self.widget)
        QVBoxLayout(self.group)
        self.group.setVisible(not self.field.collapsed)
        fold = QPushButton(
            "Expand" if self.field.collapsed else "Collapse", self.widget
        )
        header.addWidget(fold)
        fold.clicked.connect(lambda: self._fold(fold))
        self.layout.addWidget(self.group)

    def _fold(self, button: QPushButton) -> None:
        if self.group is not None:
            expanded = self.group.isHidden()
            self.group.setVisible(expanded)
            button.setText("Collapse" if expanded else "Expand")

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
            self.mode = QComboBox(self.widget)
            self.mode.setObjectName(self._name("mode"))
            self.mode.addItem("Direct", "direct")
            self.mode.addItem("Expression", "expr")
            self.mode.currentIndexChanged.connect(self._mode_changed)
            header.addWidget(self.mode)
        self._input_layout = header
        header.addWidget(widget)
        connect_value_widget(widget, self._input_changed)

    def _activate(self) -> None:
        if not self._displaying and self.enabled is not None:
            self._changed(self.field.path, {} if self.enabled.isChecked() else None)

    def _mode_changed(self, _index: int) -> None:
        if self._displaying:
            return
        if self.field.options is not None:
            text = (
                self.input.text()
                if isinstance(self.input, QLineEdit)
                else str(self.field.raw)
            )
            self._displaying = True
            try:
                expression = self.mode is not None and self.mode.currentIndex() == 1
                self._select_input(expression=expression)
                if isinstance(self.input, QLineEdit):
                    self.input.setText(text)
                else:
                    self._display_choices(
                        self.field.value if self.field.has_value else self.field.raw
                    )
            finally:
                self._displaying = False
        self._input_changed()

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
            if index >= 0 and self.field.options is not None:
                self._changed(self.field.path, self.field.options[index])
        elif isinstance(widget, QLineEdit):
            expression = self.mode is not None and self.mode.currentIndex() == 1
            text = widget.text()
            try:
                parsed = _parse(self.field, text, expression=expression)
            except ValueError as exc:
                self._unparsed(self.field.path, LocalText(text, str(exc), expression))
                return
            self._changed(self.field.path, parsed)

    def display(
        self, field: FieldView, local: LocalText | None, *, editable: bool
    ) -> None:
        """Show a same-shape publication, keeping local text and blocking writes.

        editable controls input enablement, not model validity. local overrides
        parsed text/mode and hides its evaluated label. Shape changes require
        a new row; metadata/folding remain initially declared presentation.
        """
        self.field = field
        self._displaying = True
        try:
            self._display_input(local, editable=editable)
            self.value.setText(
                str(field.value) if field.has_value and local is None else ""
            )
        finally:
            self._displaying = False

    def _display_input(self, local: LocalText | None, *, editable: bool) -> None:
        field = self.field
        if self.enabled is not None:
            self.enabled.setChecked(field.active)
            self.enabled.setEnabled(editable)
        if self.mode is not None:
            self.mode.setCurrentIndex(
                1 if (local.expression if local else _expression(field)) else 0
            )
            self.mode.setEnabled(editable)
        self._select_input(expression=local.expression if local else _expression(field))
        widget = self.input
        if widget is None:
            return
        widget.setEnabled(editable)
        if isinstance(widget, QComboBox):
            self._display_choices(field.raw)
        elif isinstance(widget, QLineEdit):
            text = local.text if local else _input_text(field)
            # Avoid resetting cursor/selection on unchanged publications.
            if widget.text() != text:
                write_value_widget(widget, text)
        elif isinstance(widget, QCheckBox):
            write_value_widget(widget, field.raw)
