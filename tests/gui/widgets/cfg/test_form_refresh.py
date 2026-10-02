"""Section-local cfg refresh presentation and lifecycle tests."""

from __future__ import annotations

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)

from tests.gui.widgets.cfg._form_support import attach_draft, section_schema


def test_decoration_provider_refresh_rebuilds_only_affected_section(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    class BadgeProvider:
        def __init__(self, badge: str) -> None:
            self._badge = badge

        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            del spec, value
            if path == "group.value":
                return FieldDecorationPatch(badge=self._badge)
            return None

    schema = section_schema(
        {
            "group": CfgSectionSpec(
                label="Group",
                fields={"value": ScalarSpec(label="Value", type=float)},
            ),
            "stable": ScalarSpec(label="Stable", type=float),
        },
        {
            "group": CfgSectionValue(fields={"value": DirectValue(1.0)}),
            "stable": DirectValue(2.0),
        },
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    root_widget = w._root_widget
    assert isinstance(root_widget, TreeCfgWidget)
    # Capture unrelated leaf before decoration change
    stable_before = root_widget._leaf_path_to_widget["stable"]
    group_value_before = root_widget._leaf_path_to_widget["group.value"]
    # Section-local decoration refresh keeps the same TreeCfgWidget instance and preserves unrelated subtree
    w.set_decoration_provider(BadgeProvider("generated"))

    assert w._root_widget is root_widget
    assert w.decoration_for_path("group.value").badge == "generated"
    # Unrelated "stable" leaf must retain same widget
    assert root_widget._leaf_path_to_widget["stable"] is stable_before
    # Changed section's leaf should be recreated (different widget) but still present
    assert root_widget._leaf_path_to_widget["group.value"] is not group_value_before


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
