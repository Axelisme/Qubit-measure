"""Scalar value controls shared by FormEditor and analysis forms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from qtpy.QtGui import QDoubleValidator, QIntValidator
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox

FIELD_INPUT_MIN_WIDTH = 20


def make_value_widget(
    type_: type,
    default: Any,
    choices: Sequence[object] | None,
    *,
    editable: bool = True,
    decimals: int | None = None,
    optional: bool = False,
) -> QWidget:
    """Build an input widget from raw field attributes."""
    if optional:
        # An optional scalar may be empty (= None), which a spinbox cannot show.
        # Render a QLineEdit: empty text = None, a numeric validator keeps input
        # well-formed. choices/bool optionals are not supported (fast-fail).
        if choices or type_ is bool:
            raise RuntimeError(
                "optional ScalarSpec does not support choices/bool widgets"
            )
        w = QLineEdit("" if default in (None, "") else str(default))
        w.setPlaceholderText("(none)")
        if type_ is int:
            w.setValidator(QIntValidator())
        elif type_ is float:
            w.setValidator(QDoubleValidator())
        w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        w.setEnabled(editable)
        return w
    if choices is not None:
        w = QComboBox()
        w.addItems([str(c) for c in choices])
        idx = w.findText(str(default))
        if idx >= 0:
            w.setCurrentIndex(idx)
        else:
            w.setCurrentIndex(-1)
            if hasattr(w, "setPlaceholderText"):
                w.setPlaceholderText("Select...")
        w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        w.setEnabled(editable)
        return w
    if type_ is bool:
        w = QCheckBox()
        w.setChecked(bool(default))
        w.setEnabled(editable)
        return w
    if type_ is int:
        w = QSpinBox()
        w.setRange(-(2**31), 2**31 - 1)
        w.setValue(int(default))
        w.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        w.setEnabled(editable)
        return w
    if type_ is float:
        w = TrimDoubleSpinBox()
        w.setRange(-1e12, 1e12)
        w.setDecimals(decimals if decimals is not None else 6)
        w.setValue(float(default))
        w.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        w.setEnabled(editable)
        return w
    w = QLineEdit(str(default))
    w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
    w.setEnabled(editable)
    return w


def read_value_widget(w: QWidget, type_: type, fallback: Any = None) -> Any:
    """Read the current value from a widget created by make_value_widget."""
    if isinstance(w, QComboBox):
        if w.currentIndex() < 0:
            return fallback
        txt = w.currentText()
        return type_(txt) if type_ is not str else txt
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, (QSpinBox, TrimDoubleSpinBox)):
        return w.value()
    if isinstance(w, QLineEdit):
        return type_(w.text())
    return fallback


def write_value_widget(widget: QWidget, value: object) -> None:
    """Write a raw value to a supported scalar input widget."""
    if isinstance(widget, QComboBox):
        index = widget.findText(str(value))
        if index >= 0:
            widget.setCurrentIndex(index)
        return
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
        return
    if isinstance(widget, QSpinBox):
        # tolerated(cast): object cannot prove int() support; retain inputs/errors.
        widget.setValue(int(cast(Any, value)))
        return
    if isinstance(widget, TrimDoubleSpinBox):
        # tolerated(cast): object cannot prove float() support; retain inputs/errors.
        widget.setValue(float(cast(Any, value)))
        return
    if isinstance(widget, QLineEdit):
        widget.setText("" if value is None else str(value))
        return
    raise TypeError(f"Unsupported value widget {type(widget).__name__}")


def connect_value_widget(widget: QWidget, callback: Callable[..., object]) -> None:
    """Connect the value-change signal of a supported scalar input widget."""
    if isinstance(widget, QComboBox):
        widget.currentIndexChanged.connect(callback)
        return
    if isinstance(widget, QCheckBox):
        widget.toggled.connect(callback)
        return
    if isinstance(widget, (QSpinBox, TrimDoubleSpinBox)):
        widget.valueChanged.connect(callback)
        return
    if isinstance(widget, QLineEdit):
        widget.textChanged.connect(callback)
        return
    raise TypeError(f"Unsupported value widget {type(widget).__name__}")


def connect_committed_value_widget(
    widget: QWidget, callback: Callable[..., object]
) -> None:
    """Connect committed text edits and immediate non-text value changes."""
    if isinstance(widget, QLineEdit):
        widget.editingFinished.connect(callback)
        return
    connect_value_widget(widget, callback)
