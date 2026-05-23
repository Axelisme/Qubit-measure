from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..adapter import AnalyzeParam, analyze_params_to_raw_dict
from .fields import make_value_widget, read_value_widget
from .widgets import TrimDoubleSpinBox


class AnalyzeFormWidget(QWidget):
    params_changed: Signal = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._params: list[AnalyzeParam] = []
        self._widgets: dict[str, QWidget] = {}
        self._hydrating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._form = QFormLayout()
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setSpacing(4)
        self._form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        layout.addLayout(self._form)

    def populate(self, params: list[AnalyzeParam]) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._params = list(params)
        self._widgets.clear()

        for param in params:
            widget = make_value_widget(
                param.type,
                param.default,
                param.choices,
                editable=True,
                decimals=param.decimals,
            )
            self._form.addRow(param.label + ":", widget)
            self._widgets[param.key] = widget
            self._connect_widget(widget)

    def read_params(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for param in self._params:
            widget = self._widgets[param.key]
            values[param.key] = read_value_widget(
                widget,
                param.type,
                fallback=param.default,
            )
        return analyze_params_to_raw_dict(self._params, values)

    def is_valid(self) -> bool:
        try:
            self.read_params()
        except Exception:
            return False
        return True

    def has_params(self) -> bool:
        return bool(self._params)

    def populate_values(self, values: dict[str, object]) -> None:
        self._hydrating = True
        try:
            for param in self._params:
                if param.key not in values:
                    continue
                widget = self._widgets.get(param.key)
                if widget is None:
                    continue
                value = values[param.key]
                if isinstance(widget, QComboBox):
                    idx = widget.findText(str(value))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QSpinBox):
                    if isinstance(value, bool):
                        widget.setValue(int(value))
                    elif isinstance(value, int):
                        widget.setValue(value)
                elif isinstance(widget, TrimDoubleSpinBox):
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        widget.setValue(float(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
        finally:
            self._hydrating = False

    def _connect_widget(self, widget: QWidget) -> None:
        if isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._emit_params_changed)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self._emit_params_changed)
        elif isinstance(widget, (QSpinBox, TrimDoubleSpinBox)):
            widget.valueChanged.connect(self._emit_params_changed)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._emit_params_changed)

    def _emit_params_changed(self, *_: object) -> None:
        if self._hydrating:
            return
        self.params_changed.emit(self.read_params())
