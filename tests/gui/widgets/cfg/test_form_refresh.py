"""Section-local cfg refresh presentation and lifecycle tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit, QTreeWidget
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.binding import ReferenceField, ScalarField
from zcu_tools.gui.widgets.cfg.registry import FieldRenderContext
from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

from tests.gui.widgets.cfg._form_support import section_schema
from tests.gui.widgets.cfg._refresh_support import (
    BadgeProvider,
    RecordingRenderers,
    attached_form,
)


def test_decoration_provider_refresh_rebuilds_only_affected_section(
    qapp: QApplication, ctrl: MagicMock
) -> None:
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
    rendering = RecordingRenderers()
    with attached_form(schema, ctrl, rendering) as form:
        tree = form.findChild(QTreeWidget)
        assert tree is not None
        group_item = tree.topLevelItem(0)
        stable_item = tree.topLevelItem(1)
        assert group_item is not None
        stable_widget = rendering.widgets["stable"][0]
        for count, badge in enumerate(("generated", "updated", "final"), start=2):
            previous = rendering.widgets["group.value"][-1]
            form.set_decoration_provider(BadgeProvider("group.value", badge))
            assert "group.value" in form.decoration_paths()
            assert form.findChild(QTreeWidget) is tree
            assert tree.topLevelItemCount() == 2
            assert tree.topLevelItem(0) is group_item
            assert tree.topLevelItem(1) is stable_item
            assert group_item.childCount() == 1
            assert rendering.widgets["stable"] == [stable_widget]
            assert len(rendering.widgets["group.value"]) == count
            assert rendering.widgets["group.value"][-1] is not previous
            assert form.decoration_for_path("group.value").badge == badge
            assert form.read_values() == schema.value


@pytest.mark.parametrize("refresh_path", ["ref.inner", "ref"])
def test_elided_singleton_survives_decoration_provider_refresh(
    qapp: QApplication, ctrl: MagicMock, refresh_path: str
) -> None:
    inner = CfgSectionSpec(
        label="Inner",
        fields={
            "gain": ScalarSpec(label="Gain", type=float),
            "freq": ScalarSpec(label="Freq", type=float),
        },
    )
    outer = CfgSectionSpec(label="Outer", fields={"inner": inner})
    schema = section_schema(
        {
            "ref": ReferenceSpec(kind="module", allowed=[outer], label="Ref"),
            "stable": ScalarSpec(label="Stable", type=float),
        },
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
            ),
            "stable": DirectValue(2.0),
        },
    )
    rendering = RecordingRenderers()
    with attached_form(schema, ctrl, rendering) as form:
        tree = form.findChild(QTreeWidget)
        root = form.findChild(TreeCfgWidget)
        assert tree is not None and root is not None
        ref_item = tree.topLevelItem(0)
        stable_item = tree.topLevelItem(1)
        assert ref_item is not None
        before = form.read_values()
        header = rendering.widgets["ref"][0]
        stable = rendering.widgets["stable"][0]
        form.set_decoration_provider(BadgeProvider("ref.inner.gain", "generated"))
        assert "ref.inner.gain" in form.decoration_paths()
        assert root.refresh_section(refresh_path)
        assert tree.topLevelItem(0) is ref_item
        assert tree.topLevelItem(1) is stable_item
        assert rendering.widgets["ref"] == [header]
        assert rendering.widgets["stable"] == [stable]
        assert tree.itemWidget(ref_item, 1) is header
        assert ref_item.childCount() == 2
        gain_item, freq_item = ref_item.child(0), ref_item.child(1)
        assert gain_item is not None and freq_item is not None
        assert "Gain" in gain_item.text(0)
        assert "Freq" in freq_item.text(0)
        assert tree.itemWidget(gain_item, 1) is rendering.widgets["ref.inner.gain"][-1]
        assert len(rendering.widgets["ref.inner.gain"]) == 3
        assert form.decoration_for_path("ref.inner.gain").badge == "generated"
        gain = rendering.fields["ref.inner.gain"]
        assert isinstance(gain, ScalarField)
        assert gain.get_value() == DirectValue(0.5)
        assert form.read_values() == before


@pytest.mark.parametrize("owner_kind", ["section", "reference"])
def test_refresh_releases_nested_editors_and_keeps_new_bindings_live(
    qapp: QApplication, ctrl: MagicMock, owner_kind: str
) -> None:
    pulse_a = CfgSectionSpec(
        label="A", fields={"gain": ScalarSpec(label="Gain", type=float)}
    )
    pulse_b = CfgSectionSpec(
        label="B", fields={"gain": ScalarSpec(label="Gain", type=float)}
    )
    group_spec = CfgSectionSpec(
        label="Group",
        fields={
            "marker": ScalarSpec(label="Marker", type=float),
            "pulse": ReferenceSpec(
                kind="module", allowed=[pulse_a, pulse_b], optional=True
            ),
        },
    )
    group_value = CfgSectionValue(
        fields={
            "marker": DirectValue(1.0),
            "pulse": ReferenceValue(
                chosen_key="<Custom:A>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.5)}),
            ),
        }
    )
    schema = section_schema(
        {
            "group": (
                group_spec
                if owner_kind == "section"
                else ReferenceSpec(kind="module", allowed=[group_spec])
            )
        },
        {
            "group": (
                group_value
                if owner_kind == "section"
                else ReferenceValue(chosen_key="<Custom:Group>", value=group_value)
            )
        },
    )
    rendering = RecordingRenderers()
    with attached_form(schema, ctrl, rendering) as form:
        old_marker = rendering.widgets["group.marker"][-1]
        old_header = rendering.widgets["group.pulse"][-1]
        old_gain = rendering.widgets["group.pulse.gain"][-1]
        old_input = old_marker.findChild(QLineEdit)
        old_combo = old_header.findChild(QComboBox)
        assert old_input is not None and old_combo is not None
        old_text, old_selection = old_input.text(), old_combo.currentText()
        destroyed: list[str] = []
        old_marker.destroyed.connect(lambda: destroyed.append("marker"))
        old_header.destroyed.connect(lambda: destroyed.append("pulse"))
        old_gain.destroyed.connect(lambda: destroyed.append("gain"))

        form.set_decoration_provider(BadgeProvider("group.marker", "generated"))
        assert "group.pulse.gain" in form.decoration_paths()
        new_marker = rendering.widgets["group.marker"][-1]
        new_header = rendering.widgets["group.pulse"][-1]
        new_input = new_marker.findChild(QLineEdit)
        new_combo = new_header.findChild(QComboBox)
        assert new_input is not None and new_combo is not None
        assert new_marker is not old_marker and new_header is not old_header
        assert old_marker.parent() is None and old_header.parent() is None

        marker_field = rendering.fields["group.marker"]
        pulse_field = rendering.fields["group.pulse"]
        assert isinstance(marker_field, ScalarField)
        assert isinstance(pulse_field, ReferenceField)
        marker_field.set_value(DirectValue(7.0))
        assert float(new_input.text()) == 7.0
        assert old_input.text() == old_text
        pulse_field.set_value(
            ReferenceValue(
                chosen_key="<Custom:B>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.8)}),
            )
        )
        assert "B" in new_combo.currentText()
        assert old_combo.currentText() == old_selection
        assert len(rendering.widgets["group.pulse.gain"]) == 3
        pulse_field.set_enabled(False)
        assert not rendering.widgets["group.pulse.gain"][-1].isEnabled()
        pulse_field.set_enabled(True)
        assert rendering.widgets["group.pulse.gain"][-1].isEnabled()
        assert form.is_valid()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sorted(destroyed) == ["gain", "marker", "pulse"]


@pytest.mark.parametrize("root_path", ["", "scope"])
def test_tree_refresh_routes_root_and_rejects_unsupported_paths(
    qapp: QApplication, ctrl: MagicMock, root_path: str
) -> None:
    schema = section_schema(
        {"value": ScalarSpec(label="Value", type=float)},
        {"value": DirectValue(2.0)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    rendering = RecordingRenderers()
    root = TreeCfgWidget(
        draft.root,
        FieldRenderContext(registry=rendering.registry, path=root_path, top_level=True),
    )
    path = f"{root_path}.value" if root_path else "value"
    try:
        tree = root.findChild(QTreeWidget)
        assert tree is not None
        item = tree.topLevelItem(0)
        editor = rendering.widgets[path][0]
        for unsupported in (path, f"{root_path}.missing", "outside.value"):
            assert not root.refresh_section(unsupported)
            assert tree.topLevelItem(0) is item
            assert rendering.widgets[path] == [editor]
        assert root.refresh_section(root_path)
        assert len(rendering.widgets[path]) == 2
        assert rendering.widgets[path][-1] is not editor
        assert tree.topLevelItemCount() == 1
        assert draft.snapshot().value == schema.value
    finally:
        root.teardown()
        root.deleteLater()
        draft.close()
