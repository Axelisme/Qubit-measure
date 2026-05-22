"""Container widgets for LiveFields (Section/ModuleRef)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...adapter import LiteralSpec, ScalarSpec
from ...live_model import LiveField, ModuleRefLiveField, SectionLiveField
from .common import BaseLiveWidget
from .registry import FieldWidgetProtocol, get_widget_cls, register_widget


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

        # For compatibility with old code that expects .form on this widget
        self.form = QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(4)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        self.body_layout.addLayout(self.form)

        if collapsed:
            self._body.setVisible(False)

    def _on_toggle(self, checked: bool) -> None:
        if self._toggle_btn:
            self._toggle_btn.setText("▼" if checked else "▶")
        self._body.setVisible(checked)


@register_widget(SectionLiveField)
class SectionWidget(BaseLiveWidget):
    def __init__(
        self,
        field: SectionLiveField,
        top_level: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(field, parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._container = _CollapsibleSection(
            field.spec.label,
            collapsible=not top_level,
            no_header=top_level and not field.spec.label,
        )
        layout.addWidget(self._container)

        self._child_widgets: Dict[str, FieldWidgetProtocol] = {}
        self._build_children()

    def _build_children(self) -> None:
        field = cast(SectionLiveField, self._field)
        for key, child_field in field.fields.items():
            spec = child_field.spec
            # Skip hidden scalar fields
            if isinstance(spec, ScalarSpec) and spec.hidden:
                continue
            if isinstance(spec, LiteralSpec) and key in {"type", "style"}:
                continue

            widget_cls = get_widget_cls(child_field)
            w = widget_cls(child_field)  # type: ignore

            label = spec.label or key
            self._container.form.addRow(f"{label}:", cast(QWidget, w))
            self._child_widgets[key] = w

    def teardown(self) -> None:
        # Recursively teardown children
        for w in self._child_widgets.values():
            w.teardown()


@register_widget(ModuleRefLiveField)
class ModuleRefWidget(BaseLiveWidget):
    def __init__(self, field: ModuleRefLiveField, parent: Optional[QWidget] = None):
        super().__init__(field, parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header: Checkbox + Combo
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        self._expand_btn = QToolButton()
        self._expand_btn.setAutoRaise(True)
        self._expand_btn.setCheckable(True)
        self._expand_btn.setChecked(True)
        self._expand_btn.setArrowType(Qt.DownArrow)  # type: ignore[attr-defined]
        self._expand_btn.clicked.connect(self._on_toggle_subsection)
        header.addWidget(self._expand_btn)

        self._combo = QComboBox()
        self._refresh_combo_items()
        self._combo.setMinimumWidth(20)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        header.addWidget(self._combo, stretch=1)
        layout.addLayout(header)

        # Sub-section container
        self._sub_container = QWidget()
        self._sub_layout = QVBoxLayout(self._sub_container)
        self._sub_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._sub_container)

        self._sub_widget: Optional[FieldWidgetProtocol] = None
        self._refresh_sub_widget()

        # Reactive sync
        field.on_change.connect(self._on_model_changed)

    def _refresh_combo_items(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()

        field = cast(ModuleRefLiveField, self._field)
        current = field.get_chosen_key()

        # 1. Custom specs from 'allowed'
        for spec in field.spec.allowed:
            label = spec.label or "Custom"
            key = f"<Custom:{label}>"
            self._combo.addItem(label, key)

        # 2. Named modules from Library if available
        ml = field.env.ctrl.get_current_ml()
        if ml:
            from ...adapter import ModuleRefSpec

            is_module = isinstance(field.spec, ModuleRefSpec)
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
        cast(ModuleRefLiveField, self._field).set_chosen_key(key)

    def _on_model_changed(self, *_: Any) -> None:
        field = cast(ModuleRefLiveField, self._field)
        key = field.get_chosen_key()
        idx = self._combo.findData(key)
        if idx >= 0 and idx != self._combo.currentIndex():
            self._combo.blockSignals(True)
            self._combo.setCurrentIndex(idx)
            self._combo.blockSignals(False)

        self._refresh_sub_widget()

    def _on_toggle_subsection(self, expanded: bool) -> None:
        self._sub_container.setVisible(expanded)
        self._expand_btn.setArrowType(  # type: ignore[attr-defined]
            Qt.DownArrow if expanded else Qt.RightArrow  # type: ignore[attr-defined]
        )

    def _sync_expand_btn(self) -> None:
        has_subsection = self._sub_widget is not None
        self._expand_btn.setVisible(has_subsection)
        self._expand_btn.setEnabled(has_subsection)
        if not has_subsection:
            self._sub_container.setVisible(False)
            return
        expanded = self._expand_btn.isChecked()
        self._sub_container.setVisible(expanded)
        self._expand_btn.setArrowType(  # type: ignore[attr-defined]
            Qt.DownArrow if expanded else Qt.RightArrow  # type: ignore[attr-defined]
        )

    def _refresh_sub_widget(self) -> None:
        field = cast(ModuleRefLiveField, self._field)
        sub_field = field.sub_field

        if self._sub_widget and self._sub_widget.field == sub_field:
            self._sync_expand_btn()
            return

        if self._sub_widget:
            self._sub_layout.removeWidget(cast(QWidget, self._sub_widget))
            cast(QWidget, self._sub_widget).deleteLater()
            self._sub_widget = None

        if sub_field:
            widget_cls = get_widget_cls(sub_field)
            w = widget_cls(sub_field)  # type: ignore
            self._sub_widget = w
            self._sub_layout.addWidget(cast(QWidget, w))
        self._sync_expand_btn()

    def teardown(self) -> None:
        self._field.on_change.disconnect(self._on_model_changed)
        if self._sub_widget:
            self._sub_widget.teardown()
