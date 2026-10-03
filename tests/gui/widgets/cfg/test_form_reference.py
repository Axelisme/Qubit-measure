"""Form reference subtree editing and value round-trip."""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QComboBox, QLineEdit
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.widgets.cfg import CfgFormWidget
from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget
from zcu_tools.resources.context import ModuleLibrary

from tests.gui.widgets.cfg._form_support import attach_draft, section_schema
from tests.gui.widgets.cfg._tree_support import tree_item, tree_widget


@pytest.mark.parametrize(
    ("kind", "reference_key", "custom_spec", "leaf_values"),
    [
        pytest.param(
            "module",
            "pulse",
            CfgSectionSpec(
                label="Pulse Shape",
                fields={
                    "type": LiteralSpec("pulse"),
                    "gain": ScalarSpec(label="Gain", type=float),
                },
            ),
            ("gain", 0.25, 0.75),
            id="module",
        ),
        pytest.param(
            "waveform",
            "waveform",
            CfgSectionSpec(
                label="Gaussian",
                fields={
                    "style": LiteralSpec("gauss"),
                    "sigma": ScalarSpec(label="Sigma", type=float),
                },
            ),
            ("sigma", 0.5, 0.125),
            id="waveform",
        ),
    ],
)
def test_custom_reference_renders_header_and_editable_leaf(
    qapp, ctrl, kind, reference_key, custom_spec, leaf_values
):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QComboBox, QLineEdit, QTreeWidget
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

    leaf_key, initial_value, edited_value = leaf_values
    chosen_key = f"<Custom:{custom_spec.label}>"
    spec = ReferenceSpec(kind=kind, label=reference_key.title(), allowed=[custom_spec])
    schema = section_schema(
        {reference_key: spec},
        {
            reference_key: ReferenceValue(
                chosen_key=chosen_key,
                value=CfgSectionValue(fields={leaf_key: DirectValue(initial_value)}),
            )
        },
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)

    ref_widget = w.findChild(ReferenceWidget)
    assert ref_widget is not None
    combo = ref_widget.findChild(QComboBox)
    assert combo is not None
    assert combo.currentText() == custom_spec.label
    assert combo.currentData() == chosen_key

    tree = w.findChild(QTreeWidget)
    assert tree is not None
    leaves = tree.findItems(
        custom_spec.fields[leaf_key].label,
        Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive,
    )
    assert len(leaves) == 1
    editor = tree.itemWidget(leaves[0], 1)
    assert editor is not None
    line = editor.findChild(QLineEdit)
    assert line is not None
    assert line.text() == str(initial_value)
    line.setText(str(edited_value))

    reference = w.read_values().fields[reference_key]
    assert isinstance(reference, ReferenceValue)
    assert reference.chosen_key == chosen_key
    leaf = reference.value.fields[leaf_key]
    assert isinstance(leaf, DirectValue)
    assert leaf.value == edited_value
    w.detach()


def test_populate_module_ref_field_round_trip(qapp, ctrl):
    allowed_spec = CfgSectionSpec(
        label="Pulse",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    schema = section_schema(
        {"mod": ReferenceSpec(kind="module", allowed=[allowed_spec], label="Module")},
        {
            "mod": ReferenceValue(
                chosen_key="<Custom:Pulse>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.5)}),
            )
        },
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        mod = form.read_values().fields["mod"]
        assert isinstance(mod, ReferenceValue)
        assert mod.chosen_key == "<Custom:Pulse>"
        gain = mod.value.fields["gain"]
        assert isinstance(gain, DirectValue) and gain.value == pytest.approx(0.5)
    finally:
        form.detach()
        form.close()
        draft.close()


def test_module_ref_edit_survives_refresh_and_can_revert(qapp, ctrl):
    ml = ModuleLibrary()
    ml.register_module(
        my_pulse={
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_length": 1.0,
            "ro_freq": 7000.0,
        }
    )
    ctrl.get_current_ml.return_value = ml
    lib_spec, lib_value = module_cfg_to_value(ml.get_module("my_pulse"))
    schema = section_schema(
        {"mod": ReferenceSpec(kind="module", allowed=[lib_spec], label="Module")},
        {"mod": ReferenceValue(chosen_key="my_pulse", value=lib_value)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        form.show()
        reference = form.findChild(ReferenceWidget)
        assert reference is not None
        combo = reference.findChild(QComboBox)
        assert combo is not None
        assert combo.currentText() == "Lib: my_pulse"

        tree_item(form, "mod").setExpanded(True)
        qapp.processEvents()
        editor = tree_widget(form).itemWidget(tree_item(form, "mod.ro_freq"), 1)
        assert editor is not None
        entry = editor.findChild(QLineEdit)
        assert entry is not None and entry.isVisible()
        assert entry.text() == "7000.0"
        entry.setText("8000.0")
        assert combo.currentText() == "Lib: my_pulse (modified)"

        ml.update_module("my_pulse", {"ro_freq": 7500.0})
        draft.refresh_references("module")
        draft.refresh_expressions()
        modified = form.read_values().fields["mod"]
        assert isinstance(modified, ReferenceValue)
        assert modified.chosen_key == "my_pulse"
        assert modified.is_overridden is True
        frequency = modified.value.fields["ro_freq"]
        assert isinstance(frequency, DirectValue) and frequency.value == 8000.0
        assert combo.currentText() == "Lib: my_pulse (modified)"

        revert_index = combo.findText("Revert to Lib: my_pulse")
        assert revert_index >= 0
        combo.setCurrentIndex(revert_index)
        reverted = form.read_values().fields["mod"]
        assert isinstance(reverted, ReferenceValue)
        assert reverted.chosen_key == "my_pulse"
        assert reverted.is_overridden is False
        frequency = reverted.value.fields["ro_freq"]
        assert isinstance(frequency, DirectValue) and frequency.value == 7500.0
        assert combo.currentText() == "Lib: my_pulse"
        assert draft.is_valid()
    finally:
        form.detach()
        form.close()
        draft.close()


def _make_optional_module_ref_schema(enabled: bool = True) -> CfgSchema:
    from zcu_tools.gui.cfg import ReferenceSpec, ReferenceValue

    inner_spec = CfgSectionSpec(
        label="Pulse",
        fields={"ch": ScalarSpec(label="Ch", type=int)},
    )
    outer_spec = CfgSectionSpec(
        fields={
            "module": ReferenceSpec(
                kind="module", allowed=[inner_spec], label="Module", optional=True
            ),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )
    if enabled:
        inner_val = CfgSectionValue(fields={"ch": DirectValue(0)})
        outer_val = CfgSectionValue(
            fields={
                "module": ReferenceValue(chosen_key="<Custom:Pulse>", value=inner_val),
                "reps": DirectValue(10),
            }
        )
    else:
        outer_val = CfgSectionValue(fields={"reps": DirectValue(10)})
    return CfgSchema(spec=outer_spec, value=outer_val)


def test_optional_module_ref_can_enable_from_none(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox, QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

    draft = MeasureCfgBindings(ctrl).new_draft(
        _make_optional_module_ref_schema(enabled=False)
    )
    form = CfgFormWidget()
    try:
        form.attach(draft)
        form.show()
        reference = form.findChild(ReferenceWidget)
        assert reference is not None
        combo = reference.findChild(QComboBox)
        assert combo is not None
        assert combo.currentText() == "None"
        assert form.read_values().fields["module"] is None
        assert draft.is_valid()

        custom_index = combo.findText("Pulse")
        assert custom_index >= 0
        combo.setCurrentIndex(custom_index)
        qapp.processEvents()
        row = tree_item(form, "module.ch")
        assert not row.isDisabled()
        editor = tree_widget(form).itemWidget(row, 1)
        assert editor is not None
        entry = editor.findChild(QLineEdit)
        assert entry is not None and entry.isEnabled() and entry.isVisible()
        entry.setText("3")
        module = form.read_values().fields["module"]
        assert isinstance(module, ReferenceValue)
        assert module.chosen_key == "<Custom:Pulse>"
        channel = module.value.fields["ch"]
        assert isinstance(channel, DirectValue)
        assert channel.value == 3
        assert draft.is_valid()
    finally:
        form.detach()
        form.close()
        draft.close()


def test_optional_module_ref_select_none_disables_sub(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox, QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

    draft = MeasureCfgBindings(ctrl).new_draft(
        _make_optional_module_ref_schema(enabled=True)
    )
    form = CfgFormWidget()
    try:
        form.attach(draft)
        form.show()
        qapp.processEvents()
        reference = form.findChild(ReferenceWidget)
        assert reference is not None
        combo = reference.findChild(QComboBox)
        assert combo is not None
        row = tree_item(form, "module.ch")
        editor = tree_widget(form).itemWidget(row, 1)
        assert editor is not None
        entry = editor.findChild(QLineEdit)
        assert entry is not None and entry.isEnabled()
        assert entry.text() == "0"
        assert not row.isDisabled()

        none_index = combo.findText("None")
        assert none_index >= 0
        combo.setCurrentIndex(none_index)
        assert combo.currentText() == "None"
        assert form.read_values().fields["module"] is None
        assert draft.is_valid()
        disabled_row = tree_item(form, "module.ch")
        assert disabled_row.isDisabled()
        disabled_editor = tree_widget(form).itemWidget(disabled_row, 1)
        assert disabled_editor is not None
        disabled_entry = disabled_editor.findChild(QLineEdit)
        assert disabled_entry is not None and not disabled_entry.isEnabled()
    finally:
        form.detach()
        form.close()
        draft.close()


def test_missing_module_reference_is_invalid_and_shows_hint(qapp, ctrl):
    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.resources.context import ModuleLibrary

    pulse_spec = CfgSectionSpec(
        label="Pulse",
        fields={
            "type": LiteralSpec("pulse"),
            "gain": ScalarSpec(label="Gain", type=float),
        },
    )
    schema = section_schema(
        {"pulse": ReferenceSpec(kind="module", label="Pulse", allowed=[pulse_spec])},
        {
            "pulse": ReferenceValue(
                chosen_key="missing_pulse",
                value=CfgSectionValue(
                    fields={"type": DirectValue("pulse"), "gain": DirectValue(0.2)}
                ),
            )
        },
    )
    ctrl.get_current_ml.return_value = ModuleLibrary()
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        form.show()
        assert not draft.is_valid()
        missing = form.read_values().fields["pulse"]
        assert isinstance(missing, ReferenceValue)
        assert missing.chosen_key == "missing_pulse"
        hint = form.findChild(QLabel, "missingRefHint")
        assert hint is not None and hint.isVisible()
        assert "missing_pulse" in hint.text()
    finally:
        form.detach()
        form.close()
        draft.close()
