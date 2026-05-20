"""Phase 15 tests — Scheme refactoring, tree-traversal readback, and dynamic forms."""

from __future__ import annotations

import sys
from typing import Any, Optional, cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSection,
    ModuleRefField,
    MultiSweepField,
    ScalarField,
    SweepField,
    schema_to_dict,
)
from zcu_tools.gui.ui.cfg_form import CfgFormWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _schema(fields: dict) -> CfgSchema:
    return CfgSchema(root=CfgSection(fields=fields))


class MockModuleCfg:
    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data


class MockModuleLibrary:
    def __init__(self):
        self.modules = {
            "pulse_a": MockModuleCfg({"type": "pulse/const", "gain": 0.5, "ch": 1}),
            "pulse_b": MockModuleCfg({"type": "pulse/const", "gain": 0.2, "ch": 2}),
        }

    def get_module(self, name: str, override: Optional[dict] = None) -> dict:
        import copy

        from zcu_tools.utils import deepupdate

        if name not in self.modules:
            raise KeyError(name)
        cfg = copy.deepcopy(self.modules[name].to_dict())
        if override:
            deepupdate(cfg, override, behavior="force")
        return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tree_traversal_round_trip(qapp):
    """Verify tree-traversal read_schema produces correct round-trip results."""
    schema = _schema(
        {
            "gain": ScalarField(value=0.5, label="Gain", type=float),
            "freq": SweepField(start=100.0, stop=200.0, expts=11),
            "nested": CfgSection(
                fields={
                    "ch": ScalarField(value=3, label="Channel", type=int),
                }
            ),
        }
    )

    w = CfgFormWidget()
    w.populate(schema)
    out = w.read_schema()

    # Verify standard values are preserved
    assert cast(ScalarField, out.root.fields["gain"]).value == pytest.approx(0.5)
    assert cast(SweepField, out.root.fields["freq"]).start == pytest.approx(100.0)
    assert cast(SweepField, out.root.fields["freq"]).stop == pytest.approx(200.0)
    assert cast(SweepField, out.root.fields["freq"]).expts == 11
    nested = cast(CfgSection, out.root.fields["nested"])
    assert cast(ScalarField, nested.fields["ch"]).value == 3


def test_module_ref_select_and_override(qapp):
    """Verify combobox selection of defined module and parameter overrides."""
    ml = MockModuleLibrary()

    # Initial schema with pulse_a active
    expanded_a = CfgSection(
        fields={
            "gain": ScalarField(label="gain", value=0.5, type=float),
            "ch": ScalarField(label="ch", value=1, type=int, editable=False),
        }
    )
    schema = _schema(
        {
            "mod": ModuleRefField(
                module_name="pulse_a",
                override={},
                inline_cfg=None,
                expanded_content=expanded_a,
                available_modules=["pulse_a", "pulse_b"],
            )
        }
    )

    w = CfgFormWidget()
    w.populate(schema, ml=cast(Any, ml))

    # Verify combobox widget has items including "<Custom>"
    root_widget = cast(Any, w._root_widget)
    combo_widget = cast(Any, root_widget._child_widgets["mod"])
    assert isinstance(combo_widget, QComboBox)
    assert combo_widget.currentText() == "pulse_a"
    assert combo_widget.count() == 3  # pulse_a, pulse_b, <Custom>

    # Verify nested widget exists for expanded_content
    container = cast(Any, combo_widget._container)
    assert container._sub_section_widget is not None
    assert "expanded_content" in container._child_widgets

    # Read back schema and verify
    out = w.read_schema()
    mod = cast(ModuleRefField, out.root.fields["mod"])
    assert mod.module_name == "pulse_a"

    # Simulate dynamic change in GUI: select pulse_b
    combo_widget.setCurrentText("pulse_b")
    # _on_module_changed runs synchronously
    assert mod.module_name == "pulse_a"  # original unchanged

    out2 = w.read_schema()
    mod2 = cast(ModuleRefField, out2.root.fields["mod"])
    assert mod2.module_name == "pulse_b"
    assert mod2.expanded_content is not None
    expanded2 = cast(CfgSection, mod2.expanded_content)
    # pulse_b has gain 0.2 and ch 2; module_cfg_to_section makes all fields editable except "type"
    assert cast(ScalarField, expanded2.fields["gain"]).value == pytest.approx(0.2)
    assert cast(ScalarField, expanded2.fields["ch"]).value == 2
    assert cast(ScalarField, expanded2.fields["ch"]).editable  # only "type" is locked

    # Simulate editing parameter (gain) inside the newly generated sub-form
    new_sub = cast(Any, container._sub_section_widget)
    gain_field_widget = new_sub._child_widgets["gain"]
    if hasattr(gain_field_widget, "setValue"):
        gain_field_widget.setValue(0.9)
    else:
        gain_field_widget.setText("0.9")

    out3 = w.read_schema()
    mod3 = cast(ModuleRefField, out3.root.fields["mod"])
    expanded3 = cast(CfgSection, mod3.expanded_content)
    assert cast(ScalarField, expanded3.fields["gain"]).value == pytest.approx(0.9)

    # Convert schema to dict using schema_to_dict and verify it has deep-merged overrides
    res_dict = schema_to_dict(out3, cast(Any, ml))
    assert res_dict["mod"] == {"type": "pulse/const", "gain": 0.9, "ch": 2}


def test_module_ref_custom_inline(qapp):
    """Verify combobox selection of '<Custom>' switches to custom_template and reads back inline_cfg."""
    ml = MockModuleLibrary()

    custom_tmpl = CfgSection(
        fields={
            "type": ScalarField(
                label="type", value="custom/pulse", type=str, editable=False
            ),
            "gain": ScalarField(label="gain", value=0.75, type=float),
            "ch": ScalarField(label="ch", value=4, type=int),
        }
    )

    schema = _schema(
        {
            "mod": ModuleRefField(
                module_name="pulse_a",
                override={},
                inline_cfg=None,
                expanded_content=None,
                available_modules=["pulse_a", "pulse_b"],
                custom_template=custom_tmpl,
            )
        }
    )

    w = CfgFormWidget()
    w.populate(schema, ml=cast(Any, ml))

    root_widget = cast(Any, w._root_widget)
    combo_widget = cast(Any, root_widget._child_widgets["mod"])
    container = cast(Any, combo_widget._container)

    # Initially, expanded_content was None, so no sub-section
    assert container._sub_section_widget is None

    # Select "<Custom>"
    combo_widget.setCurrentText("<Custom>")

    # Sub section widget should now be populated from custom_template
    assert container._sub_section_widget is not None
    out = w.read_schema()
    mod = cast(ModuleRefField, out.root.fields["mod"])
    assert mod.module_name is None
    assert mod.expanded_content is not None
    expanded = cast(CfgSection, mod.expanded_content)
    assert cast(ScalarField, expanded.fields["type"]).value == "custom/pulse"
    assert cast(ScalarField, expanded.fields["gain"]).value == pytest.approx(0.75)
    assert cast(ScalarField, expanded.fields["ch"]).value == 4

    # Edit a value in custom form
    sub_section = cast(Any, container._sub_section_widget)
    ch_widget = sub_section._child_widgets["ch"]
    if hasattr(ch_widget, "setValue"):
        ch_widget.setValue(8)
    else:
        ch_widget.setText("8")

    out2 = w.read_schema()
    mod2 = cast(ModuleRefField, out2.root.fields["mod"])
    expanded2 = cast(CfgSection, mod2.expanded_content)
    assert mod2.module_name is None
    assert cast(ScalarField, expanded2.fields["ch"]).value == 8

    # schema_to_dict should return the exact custom inline config
    res_dict = schema_to_dict(out2, cast(Any, ml))
    assert res_dict["mod"] == {"type": "custom/pulse", "gain": 0.75, "ch": 8}
