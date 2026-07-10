"""Container widgets for shared cfg binding sections and references."""

from __future__ import annotations

import logging
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

from zcu_tools.gui.cfg import ChoiceSectionSpec, DirectValue, LiteralSpec
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    ReferenceField,
    SectionField,
    SweepField,
)

from ..decoration import FieldDecorationProtocol
from ..registry import FieldRenderContext, FieldWidgetProtocol
from ._decoration import (
    apply_decoration,
    apply_widget_decoration,
    decorated_label_text,
)
from .common import (
    BaseLiveWidget,
    ElidedLabel,
)


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


class SectionWidget(BaseLiveWidget):
    def __init__(
        self,
        field: SectionField,
        *,
        context: FieldRenderContext,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(field, parent)
        self._context = context
        self._field_label_max_width = context.field_label_max_width
        self._path = context.path
        self._decoration_for_path = context.decoration_for_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        no_header = context.top_level and bool(context.path)
        self._container = _CollapsibleSection(
            field.spec.label,
            collapsible=not context.top_level,
            no_header=no_header or (context.top_level and not field.spec.label),
        )
        layout.addWidget(self._container)

        self._child_widgets: dict[str, FieldWidgetProtocol] = {}
        self._group_widgets: list[_CollapsibleSection] = []
        self._build_children()
        field.on_validity_changed.connect(self._on_validity_changed)
        self._on_validity_changed(field.is_valid())

    def _build_children(self) -> None:
        field = cast(SectionField, self._field)
        visible_keys = _choice_visible_keys(field)
        # Fields carrying a non-empty ScalarSpec.group render together under a
        # collapsed sub-header (e.g. "Advanced"), AFTER the ungrouped fields.
        # This is presentation-only: the value tree is unchanged — a grouped
        # field is still a flat leaf of this section (it lowers at top level).
        grouped: dict[
            str,
            list[tuple[str, str, FieldWidgetProtocol, FieldDecorationProtocol | None]],
        ] = {}
        for key, child_field in field.fields.items():
            if visible_keys is not None and key not in visible_keys:
                continue
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
            if isinstance(spec, LiteralSpec) and (
                decoration is None or decoration.hidden
            ):
                continue

            child_context = self._context.derive(path=child_path, top_level=False)
            w = self._context.registry.render(child_field, child_context)
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
            self._group_widgets.append(section)
            self._container.body_layout.addWidget(section)

    def refresh_section(self, path: str) -> bool:
        """Rebuild one descendant section in place.

        ``path`` is a dotted cfg value-tree path. An empty path rebuilds this
        section. The method deliberately keeps ancestor widgets alive so mode
        changes inside one ``ChoiceSectionSpec`` do not force the whole form to
        detach and reattach its caller-owned draft.
        """
        if path == self._path:
            self._rebuild_children()
            return True

        prefix = f"{self._path}." if self._path else ""
        if prefix and not path.startswith(prefix):
            return False
        remainder = path.removeprefix(prefix)
        key = remainder.split(".", 1)[0]
        child = self._child_widgets.get(key)
        if child is None:
            return False
        return child.refresh_section(path)

    def _rebuild_children(self) -> None:
        self._clear_children()
        self._build_children()

    def _clear_children(self) -> None:
        for widget in self._child_widgets.values():
            widget.teardown()
        self._child_widgets = {}
        while self._container.form.rowCount():
            self._container.form.removeRow(0)
        for section in self._group_widgets:
            self._container.body_layout.removeWidget(section)
            section.deleteLater()
        self._group_widgets = []

    def _add_field_row(
        self,
        form: QFormLayout,
        key: str,
        path: str,
        w: FieldWidgetProtocol,
        child_field: Any,
        *,
        decoration: FieldDecorationProtocol | None = None,
    ) -> None:
        if decoration is None and self._decoration_for_path is not None:
            decoration = self._decoration_for_path(path, child_field)
        if decoration is not None and decoration.hidden:
            return
        label = decorated_label_text(child_field.spec.label or key, decoration)
        widget = cast(QWidget, w)
        if isinstance(child_field, SectionField):
            apply_widget_decoration(widget, decoration)
            form.addRow(widget)
            return
        if isinstance(child_field, (SweepField, CenteredSweepField)):
            # Sweep widgets get their own full-width row; label goes on the line above
            label_widget = ElidedLabel(
                f"{label}:",
                max_width=self._field_label_max_width,
            )
            apply_decoration(label_widget, widget, decoration)
            form.addRow(label_widget)
            form.addRow(widget)
        else:
            label_widget = ElidedLabel(
                f"{label}:",
                max_width=self._field_label_max_width,
            )
            apply_decoration(label_widget, widget, decoration)
            form.addRow(label_widget, widget)

    def teardown(self) -> None:
        field = cast(SectionField, self._field)
        field.on_validity_changed.disconnect(self._on_validity_changed)
        self._clear_children()

    def _on_validity_changed(self, valid: bool) -> None:
        self._container.set_invalid(not valid)


def _choice_visible_keys(field: SectionField) -> set[str] | None:
    spec = field.spec
    if not isinstance(spec, ChoiceSectionSpec):
        return None
    visible = set(spec.fields)
    for binding in spec.bindings:
        selector = field.fields.get(binding.selector_key)
        value = selector.get_value() if selector is not None else None
        choice = str(value.value) if isinstance(value, DirectValue) else ""
        try:
            active_spec = binding.choices[choice]
        except KeyError as exc:
            expected = ", ".join(sorted(binding.choices))
            raise ValueError(
                f"ChoiceSectionSpec selector {binding.selector_key!r} has unknown "
                f"value {choice!r}; expected one of: {expected}"
            ) from exc
        active = set(active_spec.fields)
        visible -= binding.controlled_field_keys() - active
    return visible


class ReferenceWidget(BaseLiveWidget):
    def __init__(
        self,
        field: ReferenceField,
        *,
        context: FieldRenderContext,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(field, parent)
        self._context = context
        self._path = context.path

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

    _NONE_KEY = "<None>"

    def _refresh_combo_items(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()

        field = cast(ReferenceField, self._field)
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

        # 2. App-local catalog keys, already filtered to compatible labels.
        compatible = field.available_keys()
        if compatible:
            self._combo.insertSeparator(self._combo.count())
            for name in compatible:
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

    def _on_combo_changed(self, index: int) -> None:
        key = self._combo.itemData(index)
        field = cast(ReferenceField, self._field)
        if key == self._NONE_KEY:
            field.set_enabled(False)
        else:
            if field.spec.optional and not field.is_enabled:
                field.set_enabled(True)
            field.set_chosen_key(key)
            self._expand_btn.setChecked(str(key).startswith("<Custom:"))
            self._on_toggle_subsection(self._expand_btn.isChecked())

    def _should_expand_by_default(self) -> bool:
        field = cast(ReferenceField, self._field)
        return field.get_chosen_key().startswith("<Custom:")

    def _on_model_enabled_changed(self, enabled: bool) -> None:
        self._combo.blockSignals(True)
        if enabled:
            field = cast(ReferenceField, self._field)
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
        field = cast(ReferenceField, self._field)
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
        field = cast(ReferenceField, self._field)
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
        field = cast(ReferenceField, self._field)
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
            sub_context = self._context.derive(top_level=True)
            w = self._context.registry.render(sub_field, sub_context)
            self._sub_widget = w
            self._sub_layout.addWidget(cast(QWidget, w))
        self._sync_expand_btn()

    def refresh_section(self, path: str) -> bool:
        if self._sub_widget is None:
            return False
        return self._sub_widget.refresh_section(path)

    def teardown(self) -> None:
        field = cast(ReferenceField, self._field)
        field.on_change.disconnect(self._on_model_changed)
        field.on_validity_changed.disconnect(self._on_validity_changed)
        if field.spec.optional:
            field.on_enabled_changed.disconnect(self._on_model_enabled_changed)
        if self._sub_widget:
            self._sub_widget.teardown()

    def _on_validity_changed(self, valid: bool) -> None:
        field = cast(ReferenceField, self._field)
        logger.debug(
            "ReferenceWidget.validity_changed: key=%r valid=%r",
            field.get_chosen_key(),
            valid,
        )
        style = "" if valid else "border: 1px solid red;"
        self._combo.setStyleSheet(style)
        self._expand_btn.setStyleSheet("" if valid else "color: red;")
        self._refresh_missing_ref_hint()
