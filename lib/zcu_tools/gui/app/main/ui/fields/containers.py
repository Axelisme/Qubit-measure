"""Container widgets for LiveFields (Section/ModuleRef)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

logger = logging.getLogger(__name__)

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

from zcu_tools.gui.session.events import MlChangedPayload

from ...adapter import LiteralSpec
from ...live_model import (
    DeviceRefLiveField,
    ModuleRefLiveField,
    SectionLiveField,
    SweepLiveField,
)
from .common import BaseLiveWidget, ElidedLabel
from .registry import FieldWidgetProtocol, get_widget_cls, register_widget

_TONE_STYLES = {
    "muted": "color: #6b7280;",
    "info": "color: #2563eb;",
    "warning": "color: #8a5a00;",
    "error": "color: #b00020;",
}


class _CollapsibleSection(QWidget):
    """Internal helper for collapsible headers."""

    def __init__(
        self,
        label: str,
        collapsible: bool = True,
        collapsed: bool = False,
        no_header: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = None
        self._header_label: QLabel | None = None

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
                self._header_label = QLabel(f"<b>{label}</b>")
                header_row.addWidget(self._header_label, stretch=1)
                outer.addWidget(header)
            else:
                if label:
                    self._header_label = QLabel(f"<b>{label}</b>")
                    outer.addWidget(self._header_label)

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

    def set_invalid(self, invalid: bool) -> None:
        style = "color: red;" if invalid else ""
        if self._header_label is not None:
            self._header_label.setStyleSheet(style)
        if self._toggle_btn is not None:
            self._toggle_btn.setStyleSheet(style)


@register_widget(SectionLiveField)
class SectionWidget(BaseLiveWidget):
    def __init__(
        self,
        field: SectionLiveField,
        top_level: bool = False,
        no_header: bool = False,
        field_label_max_width: int | None = None,
        path: str = "",
        decoration_for_path: Callable[[str, Any], Any] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(field, parent)
        self._field_label_max_width = field_label_max_width
        self._path = path
        self._decoration_for_path = decoration_for_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._container = _CollapsibleSection(
            field.spec.label,
            collapsible=not top_level and not no_header,
            no_header=no_header or (top_level and not field.spec.label),
        )
        layout.addWidget(self._container)

        self._child_widgets: dict[str, FieldWidgetProtocol] = {}
        self._build_children()
        field.on_validity_changed.connect(self._on_validity_changed)
        self._on_validity_changed(field.is_valid())

    def _build_children(self) -> None:
        field = cast(SectionLiveField, self._field)
        # Fields carrying a non-empty ScalarSpec.group render together under a
        # collapsed sub-header (e.g. "Advanced"), AFTER the ungrouped fields.
        # This is presentation-only: the value tree is unchanged — a grouped
        # field is still a flat leaf of this section (it lowers at top level).
        grouped: dict[str, list[tuple[str, str, FieldWidgetProtocol, Any | None]]] = {}
        for key, child_field in field.fields.items():
            child_path = f"{self._path}.{key}" if self._path else key
            spec = child_field.spec
            decoration = (
                None
                if self._decoration_for_path is None
                else self._decoration_for_path(child_path, child_field)
            )
            # LiteralSpec is a fixed value with no editing degree of freedom, so
            # it has no widget — the GUI decides not to render it (the spec does
            # not carry any "visible" flag). This covers discriminators
            # (type/style) and adapter lock_literal'd fields uniformly.
            if isinstance(spec, LiteralSpec) and getattr(decoration, "hidden", True):
                continue

            widget_cls = get_widget_cls(child_field)
            if widget_cls is SectionWidget:
                w = SectionWidget(
                    cast(SectionLiveField, child_field),
                    field_label_max_width=self._field_label_max_width,
                    path=child_path,
                    decoration_for_path=self._decoration_for_path,
                )
            elif widget_cls is ModuleRefWidget:
                w = ModuleRefWidget(
                    cast(ModuleRefLiveField, child_field),
                    path=child_path,
                    decoration_for_path=self._decoration_for_path,
                    field_label_max_width=self._field_label_max_width,
                )
            else:
                w = widget_cls(child_field)  # type: ignore
            self._child_widgets[key] = w

            group = getattr(spec, "group", "") or ""
            if group:
                grouped.setdefault(group, []).append((key, child_path, w, decoration))
                continue
            self._add_field_row(
                self._container.form,
                key,
                child_path,
                w,
                child_field,
                decoration=decoration,
            )

        for group_label, entries in grouped.items():
            section = _CollapsibleSection(group_label, collapsible=True, collapsed=True)
            for key, child_path, w, decoration in entries:
                self._add_field_row(
                    section.form,
                    key,
                    child_path,
                    w,
                    field.fields[key],
                    decoration=decoration,
                )
            self._container.body_layout.addWidget(section)

    def _add_field_row(
        self,
        form: QFormLayout,
        key: str,
        path: str,
        w: FieldWidgetProtocol,
        child_field: Any,
        *,
        decoration: Any | None = None,
    ) -> None:
        if decoration is None and self._decoration_for_path is not None:
            decoration = self._decoration_for_path(path, child_field)
        if getattr(decoration, "hidden", False):
            return
        label = _decorated_label_text(child_field.spec.label or key, decoration)
        widget = cast(QWidget, w)
        if isinstance(child_field, SweepLiveField):
            # Sweep widgets get their own full-width row; label goes on the line above
            label_widget = ElidedLabel(
                f"{label}:",
                max_width=self._field_label_max_width,
            )
            _apply_decoration(label_widget, widget, decoration)
            form.addRow(label_widget)
            form.addRow(widget)
        else:
            label_widget = ElidedLabel(
                f"{label}:",
                max_width=self._field_label_max_width,
            )
            _apply_decoration(label_widget, widget, decoration)
            form.addRow(label_widget, widget)

    def teardown(self) -> None:
        field = cast(SectionLiveField, self._field)
        field.on_validity_changed.disconnect(self._on_validity_changed)
        for w in self._child_widgets.values():
            w.teardown()

    def _on_validity_changed(self, valid: bool) -> None:
        self._container.set_invalid(not valid)


@register_widget(ModuleRefLiveField)
class ModuleRefWidget(BaseLiveWidget):
    def __init__(
        self,
        field: ModuleRefLiveField,
        path: str = "",
        decoration_for_path: Callable[[str, Any], Any] | None = None,
        field_label_max_width: int | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(field, parent)
        self._path = path
        self._decoration_for_path = decoration_for_path
        self._field_label_max_width = field_label_max_width

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header: Checkbox + Combo
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        # Declare before _refresh_combo_items so _sync_expand_btn can read them
        self._sub_widget: FieldWidgetProtocol | None = None
        self._sub_container = QWidget()
        self._sub_layout = QVBoxLayout(self._sub_container)
        self._sub_layout.setContentsMargins(0, 0, 0, 0)

        self._expand_btn = QToolButton()
        self._expand_btn.setAutoRaise(True)
        self._expand_btn.setCheckable(True)
        self._expand_btn.setChecked(self._should_expand_by_default())
        self._expand_btn.setArrowType(Qt.DownArrow)  # type: ignore[attr-defined]
        self._expand_btn.clicked.connect(self._on_toggle_subsection)
        header.addWidget(self._expand_btn)

        self._combo = QComboBox()
        self._refresh_combo_items()
        self._combo.setMinimumWidth(20)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        header.addWidget(self._combo, stretch=1)
        layout.addLayout(header)

        self._missing_ref_hint = QLabel()
        self._missing_ref_hint.setObjectName("missingRefHint")
        self._missing_ref_hint.setStyleSheet("color: #b00020; font-size: 11px;")
        self._missing_ref_hint.setVisible(False)
        layout.addWidget(self._missing_ref_hint)

        layout.addWidget(self._sub_container)
        self._refresh_missing_ref_hint()
        self._refresh_sub_widget()

        # Reactive sync
        field.on_change.connect(self._on_model_changed)
        field.on_validity_changed.connect(self._on_validity_changed)
        self._on_validity_changed(field.is_valid())
        if field.spec.optional:
            field.on_enabled_changed.connect(self._on_model_enabled_changed)
            if not field.is_enabled:
                self._sub_container.setEnabled(False)

        # Subscribe to library-set changes so the combo refreshes when a new
        # module/waveform is added — model.on_change only fires when *this
        # field's referenced entry* changes, not when the library grows.
        field.env.bus.subscribe(MlChangedPayload, self._on_ml_changed)

    _NONE_KEY = "<None>"

    def _refresh_combo_items(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()

        field = cast(ModuleRefLiveField, self._field)
        current = field.get_chosen_key()

        # 0. None option for optional fields
        if field.spec.optional:
            self._combo.addItem("None", self._NONE_KEY)
            self._combo.insertSeparator(self._combo.count())

        # 1. Custom specs from 'allowed'
        for spec in field.spec.allowed:
            label = spec.label or "Custom"
            key = f"<Custom:{label}>"
            self._combo.addItem(label, key)

        # 2. Named modules from Library if available, filtered to allowed spec labels
        ml = field.env.ctrl.get_current_ml()
        if ml:
            from ...adapter import ModuleRefSpec
            from ...cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

            is_module = isinstance(field.spec, ModuleRefSpec)
            store = ml.modules if is_module else ml.waveforms
            allowed_labels = {s.label for s in field.spec.allowed}
            resolve_fn = module_cfg_to_value if is_module else waveform_cfg_to_value
            compatible: list[str] = []
            for name, cfg in store.items():
                try:
                    entry_spec, _ = resolve_fn(cfg)
                except Exception:
                    # A malformed library entry is skipped from this dropdown
                    # (it can't be offered as a choice) but must not silently
                    # vanish — log it so a broken/corrupt entry is discoverable.
                    logger.warning(
                        "ModuleRef combo: skipping unresolvable library entry %r",
                        name,
                        exc_info=True,
                    )
                    continue
                if entry_spec.label in allowed_labels:
                    compatible.append(name)
            if compatible:
                self._combo.insertSeparator(self._combo.count())
                for name in sorted(compatible):
                    if name == current and field.is_modified():
                        self._combo.addItem(f"Lib: {name} (modified)", name)
                        self._combo.addItem(f"Revert to Lib: {name}", name)
                    else:
                        self._combo.addItem(f"Lib: {name}", name)

        if field.spec.optional and not field.is_enabled:
            self._combo.setCurrentIndex(0)  # None option
        else:
            idx = self._combo.findData(current)
            if idx < 0 and field.has_missing_library_ref():
                self._combo.addItem(f"Missing: {current}", current)
                idx = self._combo.findData(current)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)
        self._sync_expand_btn()

    def _on_ml_changed(self, _payload: MlChangedPayload) -> None:
        # Library set changed (entry added / removed / renamed): rebuild the combo
        # list so newly added modules appear without requiring a re-open.
        # _refresh_combo_items already blocks combo signals internally, so this
        # cannot accidentally re-enter _on_combo_changed.
        self._refresh_combo_items()

    def _on_combo_changed(self, index: int) -> None:
        key = self._combo.itemData(index)
        field = cast(ModuleRefLiveField, self._field)
        if key == self._NONE_KEY:
            field.set_enabled(False)
        else:
            if field.spec.optional and not field.is_enabled:
                field.set_enabled(True)
            field.set_chosen_key(key)
            self._expand_btn.setChecked(str(key).startswith("<Custom:"))
            self._on_toggle_subsection(self._expand_btn.isChecked())

    def _should_expand_by_default(self) -> bool:
        field = cast(ModuleRefLiveField, self._field)
        return field.get_chosen_key().startswith("<Custom:")

    def _on_model_enabled_changed(self, enabled: bool) -> None:
        self._combo.blockSignals(True)
        if enabled:
            field = cast(ModuleRefLiveField, self._field)
            idx = self._combo.findData(field.get_chosen_key())
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
        else:
            self._combo.setCurrentIndex(0)  # None option
        self._combo.blockSignals(False)
        self._sub_container.setEnabled(enabled)
        self._sync_expand_btn()

    def _on_model_changed(self, *_: Any) -> None:
        self._refresh_combo_items()
        self._refresh_missing_ref_hint()
        self._refresh_sub_widget()

    def _refresh_missing_ref_hint(self) -> None:
        field = cast(ModuleRefLiveField, self._field)
        if field.has_missing_library_ref():
            key = field.get_chosen_key()
            self._missing_ref_hint.setText(
                f"Missing library reference: {key}. "
                "Switch key, or re-add an entry of that name to re-link."
            )
            self._missing_ref_hint.setVisible(True)
            return
        self._missing_ref_hint.setVisible(False)

    def _on_toggle_subsection(self, expanded: bool) -> None:
        self._sub_container.setVisible(expanded)
        self._expand_btn.setArrowType(  # type: ignore[attr-defined]
            Qt.DownArrow if expanded else Qt.RightArrow  # type: ignore[attr-defined]
        )

    def _sync_expand_btn(self) -> None:
        field = cast(ModuleRefLiveField, self._field)
        has_subsection = self._sub_widget is not None
        visible = has_subsection and (not field.spec.optional or field.is_enabled)
        self._expand_btn.setVisible(visible)
        self._expand_btn.setEnabled(visible)
        if not visible:
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
            self._sub_widget.teardown()
            self._sub_layout.removeWidget(cast(QWidget, self._sub_widget))
            cast(QWidget, self._sub_widget).deleteLater()
            self._sub_widget = None

        if sub_field:
            widget_cls = get_widget_cls(sub_field)
            if widget_cls == SectionWidget:
                w = SectionWidget(
                    sub_field,
                    no_header=True,
                    field_label_max_width=self._field_label_max_width,
                    path=self._path,
                    decoration_for_path=self._decoration_for_path,
                )
            else:
                w = widget_cls(sub_field)  # type: ignore
            self._sub_widget = w
            self._sub_layout.addWidget(cast(QWidget, w))
        self._sync_expand_btn()

    def teardown(self) -> None:
        field = cast(ModuleRefLiveField, self._field)
        field.on_change.disconnect(self._on_model_changed)
        field.on_validity_changed.disconnect(self._on_validity_changed)
        if field.spec.optional:
            field.on_enabled_changed.disconnect(self._on_model_enabled_changed)
        # Mirror the subscribe in __init__; prevents stale callbacks when the
        # widget is detached or the section is re-built (e.g. discriminator switch).
        field.env.bus.unsubscribe(MlChangedPayload, self._on_ml_changed)
        if self._sub_widget:
            self._sub_widget.teardown()

    def _on_validity_changed(self, valid: bool) -> None:
        field = cast(ModuleRefLiveField, self._field)
        logger.debug(
            "ModuleRefWidget.validity_changed: key=%r valid=%r",
            field.get_chosen_key(),
            valid,
        )
        style = "" if valid else "border: 1px solid red;"
        self._combo.setStyleSheet(style)
        self._expand_btn.setStyleSheet("" if valid else "color: red;")
        self._refresh_missing_ref_hint()


def _decorated_label_text(label: str, decoration: Any | None) -> str:
    if decoration is None:
        return label
    label_suffix = getattr(decoration, "label_suffix", "")
    badge = getattr(decoration, "badge", "")
    text = f"{label}{label_suffix}"
    if badge:
        text = f"{text} [{badge}]"
    return text


def _apply_decoration(
    label_widget: QLabel,
    value_widget: QWidget,
    decoration: Any | None,
) -> None:
    if decoration is None:
        return
    enabled = bool(getattr(decoration, "enabled", True))
    label_widget.setEnabled(enabled)
    value_widget.setEnabled(enabled)
    tooltip = str(getattr(decoration, "tooltip", "") or "")
    if tooltip:
        label_widget.setToolTip(tooltip)
        value_widget.setToolTip(tooltip)
    tone = str(getattr(decoration, "tone", "normal") or "normal")
    style = _TONE_STYLES.get(tone, "")
    if style:
        label_widget.setStyleSheet(style)


@register_widget(DeviceRefLiveField)
class DeviceRefWidget(BaseLiveWidget):
    """Combo box listing registered device names for DeviceRefLiveField."""

    def __init__(
        self, field: DeviceRefLiveField, parent: QWidget | None = None
    ) -> None:
        super().__init__(field, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._combo = QComboBox()
        self._combo.setMinimumWidth(20)
        layout.addWidget(self._combo, stretch=1)
        self._refresh_combo()
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        field.on_change.connect(self._on_model_changed)
        field.on_validity_changed.connect(self._on_validity_changed)

    def _refresh_combo(self) -> None:
        field = cast(DeviceRefLiveField, self._field)
        self._combo.blockSignals(True)
        self._combo.clear()
        for name in field.env.ctrl.list_device_names():
            self._combo.addItem(name)
        idx = self._combo.findText(field.get_chosen_name())
        self._combo.setCurrentIndex(idx if idx >= 0 else -1)
        self._combo.blockSignals(False)

    def _on_combo_changed(self, _index: int) -> None:
        name = self._combo.currentText()
        cast(DeviceRefLiveField, self._field).set_chosen_name(name)

    def _on_model_changed(self, _val: object) -> None:
        self._refresh_combo()

    def _on_validity_changed(self, valid: bool) -> None:
        self._combo.setStyleSheet("" if valid else "border: 1px solid red;")

    def teardown(self) -> None:
        field = cast(DeviceRefLiveField, self._field)
        field.on_change.disconnect(self._on_model_changed)
        field.on_validity_changed.disconnect(self._on_validity_changed)
