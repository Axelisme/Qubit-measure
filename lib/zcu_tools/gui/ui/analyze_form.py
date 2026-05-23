from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from ..adapter import AnalyzeParam, analyze_params_to_raw_dict
from .fields import make_value_widget, read_value_widget


class AnalyzeFormWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._params: list[AnalyzeParam] = []
        self._widgets: dict[str, QWidget] = {}

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
