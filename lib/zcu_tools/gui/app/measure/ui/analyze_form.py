from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, get_type_hints

# tolerated: QtPy exports the selected binding Signal dynamically.
from qtpy.QtCore import Signal  # pyright: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.widgets import error_colour, secondary_colour, set_foreground
from zcu_tools.gui.widgets.form.value_widgets import (
    connect_committed_value_widget,
    make_value_widget,
    read_value_widget,
    write_value_widget,
)
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from ..adapter.analyze_params import reconstruct_params, resolve_field_info


@dataclass(frozen=True)
class _FieldBinding:
    name: str
    bare_type: type
    choices: list[Any] | None
    label: str
    decimals: int | None
    optional: bool


class AnalyzeFormWidget(QWidget):
    """Edit dataclass parameters without publishing invalid input.

    ``params_changed`` carries a valid dataclass instance on committed edits.
    ``validity_changed`` carries a bool on every committed edit, including invalid
    input, so hosts can refresh action gates without reusing the last valid params.
    Hydration does not emit either signal. Invalid text stays editable and its
    reason appears below the fields and in the affected control's tooltip.
    """

    params_changed: Signal = Signal(object)
    validity_changed: Signal = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._params_cls: type[Any] | None = None
        self._bindings: tuple[_FieldBinding, ...] = ()
        self._widgets: dict[str, QWidget] = {}
        self._hydrating = False

        # 13 px single-column aligned ledger per spec.
        font = self.font()
        font.setPixelSize(13)
        self.setFont(font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._form = QFormLayout()
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setSpacing(4)
        self._form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        layout.addLayout(self._form)
        self._error_hint = QLabel()
        self._error_hint.setObjectName("analysisParameterError")
        self._error_hint.setWordWrap(True)
        make_label_selectable(self._error_hint)
        self._error_hint.hide()
        layout.addWidget(self._error_hint)
        self._empty_hint = QLabel("This analysis has no parameters.")
        self._empty_hint.setObjectName("analysisNoParameters")
        set_foreground(self._empty_hint, secondary_colour(self))
        self._empty_hint.setWordWrap(True)
        self._empty_hint.setVisible(False)
        layout.addWidget(self._empty_hint)

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
            bare_type, choices, label, decimals, optional = resolve_field_info(
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
        self._empty_hint.setVisible(not self._bindings)

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
            if binding.optional and isinstance(widget, QLineEdit):
                # Qt validators suppress editingFinished for partial text such as
                # '-'. The form parser must receive it to show errors on commit.
                widget.setValidator(None)
            self._form.addRow(binding.label + ":", widget)
            self._widgets[binding.name] = widget
            self._connect_widget(widget)

        # Refresh the parent's cached form size even while its pane is hidden.
        self.updateGeometry()

    def clear(self) -> None:
        """Remove every field and reset the form's dataclass contract."""
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        self._bindings = ()
        self._params_cls = None
        self._error_hint.clear()
        self._error_hint.hide()
        # Without a parameter contract the form makes no claim about parameters.
        self._empty_hint.setVisible(False)

    def read_params(self) -> object:
        """Parse current text, refresh errors, and return a dataclass instance.

        Only blank optional text means None. Nonblank unparseable text is retained
        and raises ValueError with a field-labelled reason; no params are returned.
        An unpopulated form raises RuntimeError. Reading emits neither signal.
        """
        if self._params_cls is None:
            raise RuntimeError("Analyze form has not been populated")

        raw: dict[str, Any] = {}
        errors: list[str] = []
        for binding in self._bindings:
            widget = self._widgets[binding.name]
            error = ""
            try:
                raw[binding.name] = self._read_widget_value(
                    widget,
                    binding.bare_type,
                    binding.choices,
                    optional=binding.optional,
                )
            except (ValueError, TypeError):
                error = f"{binding.label}: Enter a valid {binding.bare_type.__name__} number."
                errors.append(error)
            widget.setToolTip(error)
        message = "\n".join(errors)
        self._error_hint.setText(message)
        set_foreground(self._error_hint, error_colour(self))
        self._error_hint.setVisible(bool(errors))
        if errors:
            raise ValueError(message)
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
                write_value_widget(widget, value)
        finally:
            self._hydrating = False
        self.read_params()

    def is_valid(self) -> bool:
        """Return whether the current fields parse, refreshing visible errors."""
        if self._params_cls is None:
            return False
        try:
            self.read_params()
        # tolerated: User dataclass constructors may raise any error; project it as an invalid form.
        except Exception:  # noqa: BLE001
            return False
        return True

    def has_params(self) -> bool:
        """Return whether a dataclass form is built, independently of validity.

        Invalid edits keep the form populated. False means no parameter contract
        has been installed, not that the current text failed to parse.
        """
        return self._params_cls is not None

    def _connect_widget(self, widget: QWidget) -> None:
        connect_committed_value_widget(widget, self._emit_params_changed)

    def _emit_params_changed(self, *_: object) -> None:
        if self._hydrating:
            return
        try:
            params = self.read_params()
        except ValueError:
            self.validity_changed.emit(False)
            return
        self.params_changed.emit(params)
        self.validity_changed.emit(True)

    def _read_widget_value(
        self,
        widget: QWidget,
        bare_type: type,
        choices: list[Any] | None,
        *,
        optional: bool = False,
    ) -> Any:
        if choices is not None and isinstance(widget, QComboBox):
            current = widget.currentText()
            for choice in choices:
                if str(choice) == current:
                    return choice
            raise RuntimeError(f"Unknown analyze param choice: {current!r}")
        if optional and isinstance(widget, QLineEdit):
            txt = widget.text().strip()
            if txt == "":
                return None
            return bare_type(txt)
        return read_value_widget(widget, bare_type, fallback=None)
