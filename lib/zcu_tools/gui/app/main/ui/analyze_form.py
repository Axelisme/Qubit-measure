from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, get_type_hints

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

from zcu_tools.gui.widgets.cfg.fields import make_value_widget, read_value_widget
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox

from ..adapter.analyze_params import _resolve_field_info, reconstruct_params


@dataclass(frozen=True)
class _FieldBinding:
    name: str
    bare_type: type
    choices: list[Any] | None
    label: str
    decimals: int | None
    optional: bool


class AnalyzeFormWidget(QWidget):
    params_changed: Signal = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._params_cls: type[Any] | None = None
        self._bindings: tuple[_FieldBinding, ...] = ()
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

    def populate(self, params_instance: object) -> None:
        """Build form from a dataclass instance."""
        self.sync(params_instance)

    def sync(self, params_instance: object | None) -> None:
        """Clear, rebuild, or hydrate the form to match ``params_instance``."""
        if params_instance is None:
            self.clear()
            return
        if not dataclasses.is_dataclass(params_instance) or isinstance(
            params_instance, type
        ):
            raise TypeError("AnalyzeFormWidget.sync expects a dataclass instance")

        if type(params_instance) is self._params_cls:
            self.populate_values(params_instance)
            return

        self.clear()
        self._params_cls = type(params_instance)
        hints = get_type_hints(self._params_cls, include_extras=True)
        bindings: list[_FieldBinding] = []
        for field in dataclasses.fields(params_instance):
            bare_type, choices, label, decimals, optional = _resolve_field_info(
                field, hints
            )
            bindings.append(
                _FieldBinding(
                    name=field.name,
                    bare_type=bare_type,
                    choices=choices,
                    label=label,
                    decimals=decimals,
                    optional=optional,
                )
            )
        self._bindings = tuple(bindings)

        for binding in self._bindings:
            initial = getattr(params_instance, binding.name)
            widget = make_value_widget(
                binding.bare_type,
                initial,
                binding.choices,
                editable=True,
                decimals=binding.decimals,
                optional=binding.optional,
            )
            self._form.addRow(binding.label + ":", widget)
            self._widgets[binding.name] = widget
            self._connect_widget(widget)

    def clear(self) -> None:
        """Remove every field and reset the form's dataclass contract."""
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        self._bindings = ()
        self._params_cls = None

    def read_params(self) -> object:
        """Read widgets and return a typed dataclass instance."""
        if self._params_cls is None:
            raise RuntimeError("Analyze form has not been populated")

        raw: dict[str, Any] = {}
        for binding in self._bindings:
            widget = self._widgets[binding.name]
            raw[binding.name] = self._read_widget_value(
                widget,
                binding.bare_type,
                binding.choices,
                binding.optional,
            )
        return reconstruct_params(self._params_cls, raw)

    def populate_values(self, instance: object) -> None:
        """Restore widget values from a dataclass instance of the same type."""
        if self._params_cls is None:
            raise RuntimeError("Analyze form has not been populated")
        if type(instance) is not self._params_cls:
            raise TypeError("Analyze params instance type does not match current form")

        self._hydrating = True
        try:
            for binding in self._bindings:
                widget = self._widgets.get(binding.name)
                if widget is None:
                    continue
                value = getattr(instance, binding.name)
                if isinstance(widget, QComboBox):
                    idx = widget.findText(str(value))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, TrimDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QLineEdit):
                    # An optional field's None shows as the empty "(none)" state.
                    widget.setText("" if value is None else str(value))
        finally:
            self._hydrating = False

    def is_valid(self) -> bool:
        if self._params_cls is None:
            return False
        try:
            self.read_params()
        except Exception:
            return False
        return True

    def has_params(self) -> bool:
        return self._params_cls is not None

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

    def _read_widget_value(
        self,
        widget: QWidget,
        bare_type: type,
        choices: list[Any] | None,
        optional: bool = False,
    ) -> Any:
        if choices is not None and isinstance(widget, QComboBox):
            current = widget.currentText()
            for choice in choices:
                if str(choice) == current:
                    return choice
            raise RuntimeError(f"Unknown analyze param choice: {current!r}")
        if optional and isinstance(widget, QLineEdit):
            # Empty (the "(none)" state) or a partial/invalid entry reads as None;
            # mirrors read_scalar_widget for the cfg form's optional scalars.
            txt = widget.text().strip()
            if txt == "":
                return None
            try:
                return bare_type(txt)
            except (ValueError, TypeError):
                return None
        return read_value_widget(widget, bare_type, fallback=None)
