"""ReferenceWidget consumes options and refresh notifications from ReferenceField."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QComboBox

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.binding import ReferenceField, ResolvedReference

if TYPE_CHECKING:
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

_INNER_LABEL = "readout_rf"


def _inner_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label=_INNER_LABEL,
        fields={"freq": ScalarSpec(label="Freq", type=float)},
    )


def _inner_value() -> CfgSectionValue:
    return CfgSectionValue(fields={"freq": DirectValue(1000.0)})


class _Catalog:
    def __init__(self) -> None:
        self.entries: dict[str, ResolvedReference] = {}

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        assert kind == "module"
        return tuple(
            key
            for key, resolved in self.entries.items()
            if resolved.label in allowed_labels
        )

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        assert kind == "module"
        return self.entries.get(key)


def _make_field(catalog: _Catalog) -> ReferenceField:
    spec = ReferenceSpec(kind="module", allowed=[_inner_spec()])
    value = ReferenceValue(
        chosen_key=f"<Custom:{_INNER_LABEL}>",
        value=_inner_value(),
    )
    return ReferenceField(
        spec,
        evaluate_expression=lambda expression: 0,
        provide_options=lambda source_id: (),
        references=catalog,
        initial_val=value,
    )


def _make_widget(field: ReferenceField) -> ReferenceWidget:
    from zcu_tools.gui.widgets.cfg import FieldRenderContext, default_cfg_renderers
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

    registry = default_cfg_renderers()
    widget = registry.render(field, FieldRenderContext(registry=registry))
    assert isinstance(widget, ReferenceWidget)
    return widget


def test_module_ref_widget_combo_refreshes_from_field_catalog(qapp):
    catalog = _Catalog()
    field = _make_field(catalog)
    widget = _make_widget(field)
    try:
        combo = widget.findChild(QComboBox)
        assert combo is not None
        assert combo.currentData() == f"<Custom:{_INNER_LABEL}>"

        catalog.entries["my_module"] = ResolvedReference(_INNER_LABEL, _inner_value())
        field.refresh_references("module")

        assert [
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
            if combo.itemData(index) is not None
        ] == [
            (_INNER_LABEL, f"<Custom:{_INNER_LABEL}>"),
            ("Lib: my_module", "my_module"),
        ]
        assert combo.currentData() == f"<Custom:{_INNER_LABEL}>"
    finally:
        widget.teardown()


def test_module_ref_widget_teardown_stops_catalog_display_updates(qapp):
    catalog = _Catalog()
    field = _make_field(catalog)
    detached = _make_widget(field)
    attached = _make_widget(field)
    detached.teardown()
    try:
        detached_combo = detached.findChild(QComboBox)
        attached_combo = attached.findChild(QComboBox)
        assert detached_combo is not None
        assert attached_combo is not None
        assert detached_combo.count() == attached_combo.count() == 1

        catalog.entries["my_module"] = ResolvedReference(_INNER_LABEL, _inner_value())
        field.refresh_references("module")

        assert field.available_keys() == ("my_module",)
        assert attached_combo.findData("my_module") >= 0
        assert detached_combo.count() == 1
        assert detached_combo.currentText() == _INNER_LABEL
        assert detached_combo.currentData() == f"<Custom:{_INNER_LABEL}>"
    finally:
        attached.teardown()


def test_module_ref_widget_initial_combo_without_catalog_keys(qapp):
    widget = _make_widget(_make_field(_Catalog()))

    try:
        combo = widget.findChild(QComboBox)
        assert combo is not None
        assert combo.count() == 1
        assert combo.currentText() == _INNER_LABEL
        assert combo.currentData() == f"<Custom:{_INNER_LABEL}>"
    finally:
        widget.teardown()
