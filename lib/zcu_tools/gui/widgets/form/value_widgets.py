"""Scalar value controls shared by FormEditor and analysis forms."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtGui import QDoubleValidator, QFocusEvent, QIntValidator, QKeyEvent
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
        widget.setValue(int(cast(Any, value)))
        return
    if isinstance(widget, TrimDoubleSpinBox):
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


class _TextCommit(QObject):
    """Own one text input's idle timer and merge explicit commit notifications.

    The widget owns this filter and its timer. Hiding, reparenting, disabling,
    read-only changes and programmatic text replacement cancel pending expiry.
    Cancelling does not discard the text: a later normal focus loss may commit
    it. Signal-blocked focus loss (fold/retirement) never commits.
    """

    # Relay through Qt so native value-taking and no-argument slots both work.
    _numeric_committed = pyqtSignal(object)

    def __init__(
        self,
        widget: QLineEdit | QSpinBox | TrimDoubleSpinBox,
        text: QLineEdit,
        callback: Callable[..., object],
    ) -> None:
        super().__init__(widget)
        self._widget = widget
        self._text = text
        self._callback = callback
        self._last_text = text.text()
        self._pending_text: str | None = None
        self._interpreting = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._expired)
        text.textEdited.connect(self._edited)
        text.textChanged.connect(self._text_changed)
        widget.editingFinished.connect(self._finish)
        widget.installEventFilter(self)
        if isinstance(widget, (QSpinBox, TrimDoubleSpinBox)):
            text.installEventFilter(self)
            widget.setKeyboardTracking(False)
            self._numeric_committed.connect(callback)
            widget.valueChanged.connect(self._numeric_changed)

    def _cancel(self) -> None:
        self._timer.stop()
        self._pending_text = None

    def _editable(self) -> bool:
        return self._widget.isEnabled() and not self._widget.isReadOnly()

    def _edited(self, text: str) -> None:
        if self._editable():
            self._pending_text = text
            self._timer.start()

    def _text_changed(self, text: str) -> None:
        # textEdited precedes textChanged for a user edit. Host publications
        # and setText emit only textChanged and must not leave an old timer.
        if text != self._pending_text:
            self._cancel()

    def _expired(self) -> None:
        if self._pending_text is not None:
            self._finish()

    def _finish(self) -> None:
        self._cancel()
        if (
            not self._editable()
            or self._widget.signalsBlocked()
            or self._text.signalsBlocked()
            or self._text.text() == self._last_text
        ):
            return
        if isinstance(self._widget, (QSpinBox, TrimDoubleSpinBox)):
            self._interpreting = True
            try:
                self._widget.interpretText()
            finally:
                self._interpreting = False
        self._last_text = self._text.text()
        if isinstance(self._widget, (QSpinBox, TrimDoubleSpinBox)):
            self._numeric_committed.emit(self._widget.value())
        else:
            self._callback()

    def _numeric_changed(self, value: int | float) -> None:
        if not self._interpreting:
            self._cancel()
            self._last_text = self._text.text()
            self._numeric_committed.emit(value)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Cancel invalid lifetimes and commit ordinary focus/Enter immediately."""
        event = a1
        if a0 is None or event is None:
            return False
        if event.type() in (
            QEvent.Type.Hide,
            QEvent.Type.ParentChange,
            QEvent.Type.EnabledChange,
            QEvent.Type.ReadOnlyChange,
        ):
            self._cancel()
        elif isinstance(event, QFocusEvent) and event.type() == QEvent.Type.FocusOut:
            if event.reason() != Qt.FocusReason.PopupFocusReason:
                self._finish()
        elif (
            isinstance(event, QKeyEvent)
            and event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            # Validators may suppress editingFinished for partial input.
            self._finish()
        return False


def connect_committed_value_widget(
    widget: QWidget, callback: Callable[..., object]
) -> None:
    """Commit user text after 500 ms idle, Enter or ordinary focus loss.

    Line-edit commits call callback with no arguments. New input restarts the
    idle interval; explicit commits consume it. Programmatic text changes do not
    schedule commits. Hidden, reparented, disabled or read-only controls cancel
    pending expiry without discarding text. The widget owns the connection.
    Spinbox text uses the same policy. Spinboxes deliver their numeric value
    through Qt, preserving its adaptation for no-argument callbacks. Stepping,
    checkboxes and dropdowns keep immediate value-change callbacks.
    """
    if isinstance(widget, QLineEdit):
        _TextCommit(widget, widget, callback)
        return
    if isinstance(widget, (QSpinBox, TrimDoubleSpinBox)):
        text = widget.findChild(QLineEdit)
        _TextCommit(widget, text, callback)
        return
    connect_value_widget(widget, callback)
