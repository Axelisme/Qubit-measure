"""Reference tree folding and singleton section rendering tests."""

from __future__ import annotations

from typing import cast

import pytest
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.binding import ReferenceField

from tests.gui.widgets.cfg._form_support import attach_draft, section_schema

# ---------------------------------------------------------------------------
# cfg-tree-folding-corrections: S1 folding defaults and S2 singleton elision
# ---------------------------------------------------------------------------


def _ref_item(root, path: str):
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    assert isinstance(root, TreeCfgWidget)
    item = root._path_to_item.get(path)
    assert item is not None, f"no tree item for path {path!r}"
    return item


def test_library_reference_starts_collapsed_custom_starts_expanded(qapp, ctrl):
    """A1: library refs start collapsed, custom refs start expanded."""
    from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget
    from zcu_tools.resources.context import ModuleLibrary

    # Library case: ML contains a Direct Readout module, reference points at it
    ml = ModuleLibrary()
    ml.modules["lib_ro"] = {"type": "readout/direct", "ro_freq": 7000.0}  # type: ignore[assignment]
    ctrl.get_current_ml.return_value = ml
    lib_spec, lib_val = module_cfg_to_value(
        {"type": "readout/direct", "ro_freq": 7000.0}
    )
    schema_lib = section_schema(
        {"mod": ReferenceSpec(kind="module", allowed=[lib_spec], label="Mod")},
        {"mod": ReferenceValue(chosen_key="lib_ro", value=lib_val)},
    )
    form_lib = CfgFormWidget()
    attach_draft(form_lib, schema_lib, ctrl)
    root_lib = form_lib._root_widget
    assert isinstance(root_lib, TreeCfgWidget)
    item_lib = _ref_item(root_lib, "mod")
    assert item_lib.isExpanded() is False, "library reference should start collapsed"

    # Custom case: same shape but custom key
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
    root_custom = form_custom._root_widget
    assert isinstance(root_custom, TreeCfgWidget)
    item_custom = _ref_item(root_custom, "mod")
    assert item_custom.isExpanded() is True, "custom reference should start expanded"


def test_reference_identity_change_reapplies_folding(qapp, ctrl):
    """A1: changing reference identity reapplies library/custom folding policy."""
    from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget
    from zcu_tools.resources.context import ModuleLibrary

    ml = ModuleLibrary()
    ml.modules["lib_ro"] = {"type": "readout/direct", "ro_freq": 7000.0}  # type: ignore[assignment]
    ctrl.get_current_ml.return_value = ml
    lib_spec, lib_val = module_cfg_to_value(
        {"type": "readout/direct", "ro_freq": 7000.0}
    )
    schema = section_schema(
        {"mod": ReferenceSpec(kind="module", allowed=[lib_spec], label="Mod")},
        {"mod": ReferenceValue(chosen_key="lib_ro", value=lib_val)},
    )
    form = CfgFormWidget()
    attach_draft(form, schema, ctrl)
    root = form._root_widget
    assert isinstance(root, TreeCfgWidget)
    item = _ref_item(root, "mod")
    assert item.isExpanded() is False
    ref_widget = form.findChild(ReferenceWidget)
    assert ref_widget is not None
    field = cast(ReferenceField, ref_widget._field)
    # Switch to custom via field API (mirrors combo selection)
    field.set_chosen_key("<Custom:Direct Readout>")
    qapp.processEvents()
    assert item.isExpanded() is True
    # Switch back to library
    field.set_chosen_key("lib_ro")
    qapp.processEvents()
    assert item.isExpanded() is False


def test_disabled_optional_reference_collapsed_and_whole_row_click_does_not_expand(
    qapp, ctrl
):
    """A2: selecting None collapses and whole-row clicks cannot expand until re-enabled."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

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
    root = form._root_widget
    assert isinstance(root, TreeCfgWidget)
    item = _ref_item(root, "ref")
    assert item.isExpanded() is True
    ref_widget = form.findChild(ReferenceWidget)
    assert ref_widget is not None
    field = cast(ReferenceField, ref_widget._field)
    # Select None via shipped combo
    none_idx = ref_widget._combo.findData(ReferenceWidget._NONE_KEY)
    assert none_idx >= 0
    ref_widget._combo.setCurrentIndex(none_idx)
    qapp.processEvents()
    assert field.is_enabled is False
    assert item.isExpanded() is False
    # Whole-row click must not expand
    root._on_item_clicked(item, 0)
    qapp.processEvents()
    assert item.isExpanded() is False
    # Also direct click handler should not expand via itemClicked
    item.setExpanded(True)
    qapp.processEvents()
    # The widget should enforce collapsed for disabled optional
    assert item.isExpanded() is False
    # Re-enable via combo custom selection
    custom_idx = ref_widget._combo.findData("<Custom:Inner>")
    assert custom_idx >= 0
    ref_widget._combo.setCurrentIndex(custom_idx)
    qapp.processEvents()
    assert field.is_enabled is True
    assert item.isExpanded() is True
    # Now click should toggle
    root._on_item_clicked(item, 0)
    assert item.isExpanded() is False
    root._on_item_clicked(item, 0)
    assert item.isExpanded() is True


def test_optional_reference_initially_disabled_starts_collapsed_and_non_foldable(
    qapp, ctrl
):
    """A2: a rendered disabled optional reference starts collapsed and non-foldable."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

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
    root = form._root_widget
    assert isinstance(root, TreeCfgWidget)
    item = _ref_item(root, "ref")
    assert item.isExpanded() is False
    root._on_item_clicked(item, 0)
    assert item.isExpanded() is False


def test_singleton_nested_section_elided_and_multi_child_distinct(qapp, ctrl):
    """A3: singleton nested section elides wrapper row; multi-child does not."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

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
    # Singleton case
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
    root_single = form_single._root_widget
    assert isinstance(root_single, TreeCfgWidget)
    ref_item_single = _ref_item(root_single, "ref")
    # No redundant section row for "ref.inner"
    assert "ref.inner" not in root_single._path_to_item, (
        "singleton nested section row should be elided"
    )
    # Leaves are directly under reference without wrapper
    assert "ref.inner.gain" in root_single._leaf_path_to_widget
    assert "ref.inner.freq" in root_single._leaf_path_to_widget
    # Parent of leaf items should be ref_item, not an intermediate section item
    leaf_gain_item = None
    for idx in range(ref_item_single.childCount()):
        child = ref_item_single.child(idx)
        if child is None:
            continue
        if child.data(0, 0x0100) == "ref.inner.gain":  # UserRole
            leaf_gain_item = child
            break
    assert leaf_gain_item is not None, (
        "gain leaf should be direct child of reference after elision"
    )
    # Verify cfg path preserved via read_values
    out_single = form_single.read_values()
    ref_val = out_single.fields["ref"]
    assert isinstance(ref_val, ReferenceValue)
    assert ref_val.value.fields["inner"].fields["gain"].value == pytest.approx(0.5)  # type: ignore[union-attr]
    # Multi-child case should keep distinct structure (wrapper row present)
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
    root_multi = form_multi._root_widget
    assert isinstance(root_multi, TreeCfgWidget)
    ref_item_multi = _ref_item(root_multi, "ref")
    assert "ref.inner" in root_multi._path_to_item, (
        "multi-child shape must keep nested section row"
    )
    inner_item = root_multi._path_to_item["ref.inner"]
    assert inner_item.parent() is ref_item_multi
    assert "ref.inner.gain" in root_multi._leaf_path_to_widget
    # Changing leaf under elided singleton still round-trips
    assert "ref.gain" in root_multi._leaf_path_to_widget


def test_singleton_elision_does_not_hide_editable_reference_field(qapp, ctrl):
    """Decision stop guard: singleton elision elides only SectionField wrapper, not ReferenceField."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

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
    root = form._root_widget
    assert isinstance(root, TreeCfgWidget)
    # The singleton child is ReferenceField, not SectionField -> should NOT elide
    # The tree should NOT hide the editable ReferenceField row "ref.nested_ref"
    assert "ref.nested_ref" in root._path_to_item, (
        "editable ReferenceField must remain visible"
    )
    assert root._path_to_item["ref.nested_ref"].text(0) != ""


def test_elided_singleton_survives_decoration_provider_refresh(qapp, ctrl):
    """Blocker 1: elided singleton stays elided after section-local decoration refresh."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget, FieldDecorationPatch
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

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
    form = CfgFormWidget()
    attach_draft(form, schema, ctrl)
    root = form._root_widget
    assert isinstance(root, TreeCfgWidget)
    # Initially elided
    assert "ref.inner" not in root._path_to_item
    assert "ref.inner.gain" in root._leaf_path_to_widget
    ref_item = root._path_to_item["ref"]
    # Verify gain leaf is direct child of ref
    found_gain = False
    for idx in range(ref_item.childCount()):
        ch = ref_item.child(idx)
        if ch is not None and ch.data(0, 0x0100) == "ref.inner.gain":
            found_gain = True
            break
    assert found_gain

    # Provider that changes decoration for a leaf under the elided wrapper
    class LeafBadgeProvider:
        def decoration_for(self, path, spec, value):
            if path == "ref.inner.gain":
                return FieldDecorationPatch(badge="generated")
            return None

    form.set_decoration_provider(LeafBadgeProvider())
    # Flush pending section refresh (queued via QTimer.singleShot 0)
    qapp.processEvents()
    # Process the queued refresh
    try:
        # CfgFormWidget queues refresh via singleShot, need extra process
        form._flush_pending_section_refresh()
    except Exception:
        pass
    qapp.processEvents()
    # Still elided
    assert "ref.inner" not in root._path_to_item, (
        "wrapper should remain elided after decoration refresh"
    )
    assert "ref.inner.gain" in root._leaf_path_to_widget
    # Parentage still direct
    found_gain_after = False
    for idx in range(ref_item.childCount()):
        ch = ref_item.child(idx)
        if ch is not None and ch.data(0, 0x0100) == "ref.inner.gain":
            found_gain_after = True
            break
    assert found_gain_after
    # cfg path preserved
    out = form.read_values()
    assert out.fields["ref"].value.fields["inner"].fields["gain"].value == 0.5  # type: ignore[union-attr]


def test_singleton_wrapper_with_disabled_decoration_not_elided(qapp, ctrl):
    """Blocker 2: singleton wrapper with disabled decoration is not elided and leaves are disabled."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget, FieldDecorationPatch
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

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
        def decoration_for(self, path, spec, value):
            if path == "ref.inner":
                return FieldDecorationPatch(
                    enabled=False, badge="muted", tooltip="disabled wrapper"
                )
            return None

    form = CfgFormWidget(decoration_provider=DisabledWrapperProvider())
    attach_draft(form, schema, ctrl)
    root = form._root_widget
    assert isinstance(root, TreeCfgWidget)
    # Wrapper should NOT be elided because it has non-neutral decoration
    assert "ref.inner" in root._path_to_item, "disabled wrapper must remain visible"
    wrapper_item = root._path_to_item["ref.inner"]
    # Leaves should be under wrapper, not directly under ref
    ref_item = root._path_to_item["ref"]
    # ref should have wrapper as child, not leaves directly
    assert wrapper_item.parent() is ref_item
    # Leaves should be disabled via wrapper propagation
    gain_widget = root._leaf_path_to_widget.get("ref.inner.gain")
    assert gain_widget is not None
    # Check item disabled and widget disabled
    gain_item = None
    for idx in range(wrapper_item.childCount()):
        ch = wrapper_item.child(idx)
        if ch is not None and ch.data(0, 0x0100) == "ref.inner.gain":
            gain_item = ch
            break
    assert gain_item is not None
    assert gain_item.isDisabled() is True
    assert not gain_widget.isEnabled()  # type: ignore[attr-defined]
    # Wrapper decoration badge/tooltip should be applied (observable)
    assert wrapper_item.toolTip(0) == "disabled wrapper"
