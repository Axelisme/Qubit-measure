"""ReferenceWidget consumes options and refresh notifications from ReferenceField."""

from __future__ import annotations

from collections.abc import Sequence

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.binding import ReferenceField, ResolvedReference

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


def test_module_ref_widget_combo_refreshes_from_field_catalog(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.ui.fields.containers import ReferenceWidget

    catalog = _Catalog()
    field = _make_field(catalog)
    widget = ReferenceWidget(field)
    count_before = widget._combo.count()

    catalog.entries["my_module"] = ResolvedReference(_INNER_LABEL, _inner_value())
    field.refresh_references("module")

    assert widget._combo.count() > count_before
    assert "Lib: my_module" in [
        widget._combo.itemText(index) for index in range(widget._combo.count())
    ]
    widget.teardown()


def test_module_ref_widget_teardown_disconnects_field_callbacks(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.ui.fields.containers import ReferenceWidget

    field = _make_field(_Catalog())
    widget = ReferenceWidget(field)
    widget.teardown()

    assert widget._on_model_changed not in field.on_change._callbacks


def test_module_ref_widget_initial_combo_without_catalog_keys(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.ui.fields.containers import ReferenceWidget

    widget = ReferenceWidget(_make_field(_Catalog()))

    assert [
        widget._combo.itemText(index) for index in range(widget._combo.count())
    ] == [_INNER_LABEL]
    widget.teardown()
