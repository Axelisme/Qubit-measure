"""Reference tree folding and singleton section rendering tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QComboBox, QTreeWidget, QTreeWidgetItem
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.widgets.cfg import CfgFormWidget, FieldDecorationPatch
from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget
from zcu_tools.resources.context import ModuleLibrary

from tests.gui.widgets.cfg._form_support import attach_draft, section_schema


def _tree(form: CfgFormWidget) -> QTreeWidget:
    tree = form.findChild(QTreeWidget)
    assert tree is not None
    return tree


def _item(form: CfgFormWidget, path: str) -> QTreeWidgetItem:
    pending = [_tree(form).invisibleRootItem()]
    while pending:
        item = pending.pop()
        if item.data(0, Qt.ItemDataRole.UserRole) == path:
            return item
        for index in range(item.childCount()):
            child = item.child(index)
            if child is not None:
                pending.append(child)
    raise AssertionError(f"no tree item for path {path!r}")


def _click_row(
    qapp: QApplication, form: CfgFormWidget, item: QTreeWidgetItem
) -> None:
    form.resize(600, 400)
    form.show()
    qapp.processEvents()
    tree = _tree(form)
    tree.scrollToItem(item)
    qapp.processEvents()
    rect = tree.visualItemRect(item)
    assert rect.isValid()
    viewport = tree.viewport()
    assert viewport is not None
    position = QPoint(tree.columnWidth(0) // 2, rect.center().y())
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=position)
    qapp.processEvents()
    form.close()


def _readout_shape(ctrl: MagicMock) -> tuple[CfgSectionSpec, CfgSectionValue]:
    ml = ModuleLibrary()
    ml.register_module(
        lib_ro={
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_length": 1.0,
            "ro_freq": 7000.0,
        }
    )
    ctrl.get_current_ml.return_value = ml
    return module_cfg_to_value(ml.get_module("lib_ro"))


def test_library_reference_starts_collapsed_custom_starts_expanded(qapp, ctrl):
    lib_spec, lib_val = _readout_shape(ctrl)
    schema_lib = section_schema(
        {"mod": ReferenceSpec(kind="module", allowed=[lib_spec], label="Mod")},
        {"mod": ReferenceValue(chosen_key="lib_ro", value=lib_val)},
    )
    form_lib = CfgFormWidget()
    attach_draft(form_lib, schema_lib, ctrl)
    assert _item(form_lib, "mod").isExpanded() is False

    schema_custom = section_schema(
        {"mod": ReferenceSpec(kind="module", allowed=[lib_spec], label="Mod")},
        {
            "mod": ReferenceValue(
                chosen_key="<Custom:Direct Readout>",
                value=lib_val,
            )
        },
    )
    form_custom = CfgFormWidget()
    attach_draft(form_custom, schema_custom, ctrl)
    assert _item(form_custom, "mod").isExpanded() is True


def test_reference_identity_change_reapplies_folding(qapp, ctrl):
    lib_spec, lib_val = _readout_shape(ctrl)
    schema = section_schema(
        {"mod": ReferenceSpec(kind="module", allowed=[lib_spec], label="Mod")},
        {"mod": ReferenceValue(chosen_key="lib_ro", value=lib_val)},
    )
    form = CfgFormWidget()
    attach_draft(form, schema, ctrl)
    item = _item(form, "mod")
    assert item.isExpanded() is False
    ref_widget = form.findChild(ReferenceWidget)
    assert ref_widget is not None
    field = ref_widget.field

    field.set_chosen_key("<Custom:Direct Readout>")
    qapp.processEvents()
    assert item.isExpanded() is True
    field.set_chosen_key("lib_ro")
    qapp.processEvents()
    assert item.isExpanded() is False


def test_disabled_optional_reference_collapsed_and_whole_row_click_does_not_expand(
    qapp, ctrl
):
    inner_spec = CfgSectionSpec(
        label="Inner",
        fields={"ch": ScalarSpec(label="Ch", type=int)},
    )
    schema = section_schema(
        {
            "ref": ReferenceSpec(
                kind="module", allowed=[inner_spec], label="Ref", optional=True
            )
        },
        {
            "ref": ReferenceValue(
                chosen_key="<Custom:Inner>",
                value=CfgSectionValue(fields={"ch": DirectValue(0)}),
            )
        },
    )
    form = CfgFormWidget()
    attach_draft(form, schema, ctrl)
    item = _item(form, "ref")
    assert item.isExpanded() is True
    ref_widget = form.findChild(ReferenceWidget)
    assert ref_widget is not None
    field = ref_widget.field
    combo = ref_widget.findChild(QComboBox)
    assert combo is not None

    none_index = combo.findText("None")
    assert none_index >= 0
    combo.setCurrentIndex(none_index)
    qapp.processEvents()
    assert field.is_enabled is False
    assert item.isExpanded() is False
    _click_row(qapp, form, item)
    assert item.isExpanded() is False
    item.setExpanded(True)
    qapp.processEvents()
    assert item.isExpanded() is False

    custom_index = combo.findText("Inner")
    assert custom_index >= 0
    combo.setCurrentIndex(custom_index)
    qapp.processEvents()
    assert field.is_enabled is True
    assert item.isExpanded() is True
    _click_row(qapp, form, item)
    assert item.isExpanded() is False
    _click_row(qapp, form, item)
    assert item.isExpanded() is True


def test_optional_reference_initially_disabled_starts_collapsed_and_non_foldable(
    qapp, ctrl
):
    inner_spec = CfgSectionSpec(
        label="Inner",
        fields={"ch": ScalarSpec(label="Ch", type=int)},
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "ref": ReferenceSpec(
                    kind="module", allowed=[inner_spec], label="Ref", optional=True
                ),
                "reps": ScalarSpec(label="Reps", type=int),
            }
        ),
        value=CfgSectionValue(fields={"reps": DirectValue(10)}),
    )
    form = CfgFormWidget()
    attach_draft(form, schema, ctrl)
    item = _item(form, "ref")
    assert item.isExpanded() is False
    _click_row(qapp, form, item)
    assert item.isExpanded() is False


def test_singleton_nested_section_elided_and_multi_child_distinct(qapp, ctrl):
    inner = CfgSectionSpec(
        label="Inner",
        fields={
            "gain": ScalarSpec(label="Gain", type=float),
            "freq": ScalarSpec(label="Freq", type=float),
        },
    )
    singleton_outer = CfgSectionSpec(
        label="Outer",
        fields={"inner": inner},
    )
    multi_outer = CfgSectionSpec(
        label="Outer",
        fields={
            "gain": ScalarSpec(label="Gain", type=float),
            "inner": inner,
        },
    )
    schema_single = section_schema(
        {"ref": ReferenceSpec(kind="module", allowed=[singleton_outer], label="Ref")},
        {
            "ref": ReferenceValue(
                chosen_key="<Custom:Outer>",
                value=CfgSectionValue(
                    fields={
                        "inner": CfgSectionValue(
                            fields={"gain": DirectValue(0.5), "freq": DirectValue(5.0)}
                        )
                    }
                ),
            )
        },
    )
    form_single = CfgFormWidget()
    attach_draft(form_single, schema_single, ctrl)
    tree_single = _tree(form_single)
    ref_item_single = _item(form_single, "ref")
    gain_item = _item(form_single, "ref.inner.gain")
    freq_item = _item(form_single, "ref.inner.freq")
    assert ref_item_single.childCount() == 2
    assert gain_item.parent() is ref_item_single
    assert freq_item.parent() is ref_item_single
    assert tree_single.itemWidget(gain_item, 1) is not None
    assert tree_single.itemWidget(freq_item, 1) is not None

    out_single = form_single.read_values()
    ref_value = out_single.fields["ref"]
    assert isinstance(ref_value, ReferenceValue)
    inner_value = ref_value.value.fields["inner"]
    assert isinstance(inner_value, CfgSectionValue)
    gain_value = inner_value.fields["gain"]
    assert isinstance(gain_value, DirectValue)
    assert gain_value.value == pytest.approx(0.5)

    schema_multi = section_schema(
        {"ref": ReferenceSpec(kind="module", allowed=[multi_outer], label="Ref")},
        {
            "ref": ReferenceValue(
                chosen_key="<Custom:Outer>",
                value=CfgSectionValue(
                    fields={
                        "gain": DirectValue(1.0),
                        "inner": CfgSectionValue(
                            fields={"gain": DirectValue(0.5), "freq": DirectValue(5.0)}
                        ),
                    }
                ),
            )
        },
    )
    form_multi = CfgFormWidget()
    attach_draft(form_multi, schema_multi, ctrl)
    tree_multi = _tree(form_multi)
    ref_item_multi = _item(form_multi, "ref")
    inner_item = _item(form_multi, "ref.inner")
    nested_gain_item = _item(form_multi, "ref.inner.gain")
    direct_gain_item = _item(form_multi, "ref.gain")
    assert inner_item.parent() is ref_item_multi
    assert nested_gain_item.parent() is inner_item
    assert direct_gain_item.parent() is ref_item_multi
    assert tree_multi.itemWidget(nested_gain_item, 1) is not None
    assert tree_multi.itemWidget(direct_gain_item, 1) is not None


def test_singleton_elision_does_not_hide_editable_reference_field(qapp, ctrl):
    nested_ref_spec = ReferenceSpec(
        kind="module",
        allowed=[
            CfgSectionSpec(
                label="Nested", fields={"gain": ScalarSpec(label="Gain", type=float)}
            )
        ],
        label="NestedRef",
    )
    outer_with_ref = CfgSectionSpec(
        label="Outer",
        fields={"nested_ref": nested_ref_spec},
    )
    schema = section_schema(
        {"ref": ReferenceSpec(kind="module", allowed=[outer_with_ref], label="Ref")},
        {
            "ref": ReferenceValue(
                chosen_key="<Custom:Outer>",
                value=CfgSectionValue(
                    fields={
                        "nested_ref": ReferenceValue(
                            chosen_key="<Custom:Nested>",
                            value=CfgSectionValue(fields={"gain": DirectValue(0.3)}),
                        )
                    }
                ),
            )
        },
    )
    form = CfgFormWidget()
    attach_draft(form, schema, ctrl)
    item = _item(form, "ref.nested_ref")
    assert item.text(0) == "NestedRef"
    assert item.parent() is _item(form, "ref")
    editor = _tree(form).itemWidget(item, 1)
    assert isinstance(editor, ReferenceWidget)
    combo = editor.findChild(QComboBox)
    assert combo is not None
    assert combo.isEnabled()
    assert combo.currentText() == "Nested"


def test_singleton_wrapper_with_disabled_decoration_not_elided(qapp, ctrl):
    inner = CfgSectionSpec(
        label="Inner",
        fields={
            "gain": ScalarSpec(label="Gain", type=float),
            "freq": ScalarSpec(label="Freq", type=float),
        },
    )
    singleton_outer = CfgSectionSpec(label="Outer", fields={"inner": inner})
    schema = section_schema(
        {"ref": ReferenceSpec(kind="module", allowed=[singleton_outer], label="Ref")},
        {
            "ref": ReferenceValue(
                chosen_key="<Custom:Outer>",
                value=CfgSectionValue(
                    fields={
                        "inner": CfgSectionValue(
                            fields={"gain": DirectValue(0.5), "freq": DirectValue(5.0)}
                        )
                    }
                ),
            )
        },
    )

    class DisabledWrapperProvider:
        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            if path == "ref.inner":
                return FieldDecorationPatch(
                    enabled=False, badge="muted", tooltip="disabled wrapper"
                )
            return None

    form = CfgFormWidget(decoration_provider=DisabledWrapperProvider())
    attach_draft(form, schema, ctrl)
    wrapper_item = _item(form, "ref.inner")
    assert wrapper_item.parent() is _item(form, "ref")
    gain_item = _item(form, "ref.inner.gain")
    assert gain_item.parent() is wrapper_item
    gain_widget = _tree(form).itemWidget(gain_item, 1)
    assert gain_widget is not None
    assert gain_item.isDisabled() is True
    assert not gain_widget.isEnabled()
    assert wrapper_item.toolTip(0) == "disabled wrapper"
