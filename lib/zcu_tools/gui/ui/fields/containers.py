"""Container widgets for LiveFields (Section/ModuleRef)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...live_model import ModuleRefLiveField, SectionLiveField
from .registry import get_widget_cls, register_widget


class _CollapsibleSection(QWidget):
    """Internal helper for collapsible headers."""

    def __init__(
        self,
        label: str,
        collapsible: bool = True,
        collapsed: bool = False,
        no_header: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = None

        if not no_header:
            if collapsible:
                header = QWidget()
                header_row = QHBoxLayout(header)
                header_row.setContentsMargins(0, 0, 0, 0)
                header_row.setSpacing(2)

                self._toggle_btn = QPushButton("▼" if not collapsed else "▶")
                self._toggle_btn.setFixedWidth(16)
                self._toggle_btn.setFlat(True)
                self._toggle_btn.setCheckable(True)
                self._toggle_btn.setChecked(not collapsed)
                self._toggle_btn.clicked.connect(self._on_toggle)
                header_row.addWidget(self._toggle_btn)
                header_row.addWidget(QLabel(f"<b>{label}</b>"), stretch=1)
                outer.addWidget(header)
            else:
                if label:
                    outer.addWidget(QLabel(f"<b>{label}</b>"))

        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(8, 2, 0, 2)
        self.body_layout.setSpacing(2)
        outer.addWidget(self._body)

        if collapsed:
            self._body.setVisible(False)

    def _on_toggle(self, checked: bool) -> None:
        if self._toggle_btn:
            self._toggle_btn.setText("▼" if checked else "▶")
        self._body.setVisible(checked)


@register_widget("SectionLiveField") # Type string used to avoid circular import if needed, but we can use real type too
class SectionWidget(QWidget):
    def __init__(
        self,
        field: SectionLiveField,
        top_level: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._field = field
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._container = _CollapsibleSection(
            field.spec.label,
            collapsible=not top_level,
            no_header=top_level and not field.spec.label,
        )
        layout.addWidget(self._container)

        self.form = QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(4)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)  # type: ignore[attr-defined]
        self._container.body_layout.addLayout(self.form)

        self._child_widgets: Dict[str, QWidget] = {}
        self._build_children()

    def _build_children(self) -> None:
        from ...adapter import ScalarSpec

        for key, child_field in self._field.fields.items():
            spec = child_field.spec
            # Skip hidden scalar fields
            if hasattr(spec, "hidden") and spec.hidden:
                continue
            
            widget_cls = get_widget_cls(child_field)
            w = widget_cls(child_field) # type: ignore
            
            label = spec.label or key
            self.form.addRow(f"{label}:", w)
            self._child_widgets[key] = w


@register_widget("ModuleRefLiveField")
class ModuleRefWidget(QWidget):
    def __init__(self, field: ModuleRefLiveField, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._field = field
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header: Checkbox + Combo
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        
        self._cb = QCheckBox()
        self._cb.setChecked(True) # In UI, ModuleRef is usually optional but we don't have 'enabled' in Spec yet
        self._cb.setVisible(False) # Hide for now until we support optional modules

        self._combo = QComboBox()
        # Initial items
        self._refresh_combo_items()
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        
        header.addWidget(self._cb)
        header.addWidget(self._combo, stretch=1)
        layout.addLayout(header)

        # Sub-section container
        self._sub_container = QWidget()
        self._sub_layout = QVBoxLayout(self._sub_container)
        self._sub_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._sub_container)

        self._sub_widget: Optional[QWidget] = None
        self._refresh_sub_widget()
        
        # Reactive sync
        field.on_change.connect(self._on_model_changed)

    def _refresh_combo_items(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        
        current = self._field.get_chosen_key()
        
        # 1. Custom specs from 'allowed'
        for spec in self._field.spec.allowed:
            label = spec.label or "Custom"
            key = f"<Custom:{label}>"
            self._combo.addItem(label, key)
        
        # 2. Named modules from Library if available
        ml = self._field._ml
        if ml:
            from ...adapter import ModuleRefSpec
            is_module = isinstance(self._field.spec, ModuleRefSpec)
            store = ml.modules if is_module else ml.waveforms
            if store:
                self._combo.insertSeparator(self._combo.count())
                for name in sorted(store.keys()):
                    self._combo.addItem(f"Lib: {name}", name)

        idx = self._combo.findData(current)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

    def _on_combo_changed(self, index: int) -> None:
        key = self._combo.itemData(index)
        self._field.set_chosen_key(key)

    def _on_model_changed(self, *_: Any) -> None:
        # If chosen key changed in model, update combo and sub-widget
        key = self._field.get_chosen_key()
        idx = self._combo.findData(key)
        if idx >= 0 and idx != self._combo.currentIndex():
            self._combo.blockSignals(True)
            self._combo.setCurrentIndex(idx)
            self._combo.blockSignals(False)
        
        self._refresh_sub_widget()

    def _refresh_sub_widget(self) -> None:
        sub_field = self._field.sub_field
        
        # If we already have the correct widget, do nothing (Partial Update!)
        if self._sub_widget and hasattr(self._sub_widget, "_field") and self._sub_widget._field == sub_field: # type: ignore
            return
            
        # Clean old
        if self._sub_widget:
            self._sub_layout.removeWidget(self._sub_widget)
            self._sub_widget.deleteLater()
            self._sub_widget = None

        if sub_field:
            widget_cls = get_widget_cls(sub_field)
            self._sub_widget = widget_cls(sub_field) # type: ignore
            self._sub_layout.addWidget(self._sub_widget)
