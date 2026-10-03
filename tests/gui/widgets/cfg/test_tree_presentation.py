"""Cfg tree editing, lifecycle, and effective presentation behavior."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QWidget
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgDraft,
    ReferenceField,
    SectionField,
    SweepField,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.widgets.cfg import (
    CfgFormWidget,
    FieldDecorationPatch,
    FieldDecorationProvider,
)
from zcu_tools.gui.widgets.cfg.fields import CenteredSweepWidget, SweepWidget
from zcu_tools.resources.context import MetaDict

from tests.gui.widgets.cfg._tree_support import click_row, tree_item, tree_widget


@pytest.fixture()
def ctrl() -> MagicMock:
    controller = MagicMock()
    controller.get_bus.return_value = EventBus()
    controller.get_current_md.return_value = MetaDict()
    controller.get_current_ml.return_value = MagicMock()
    controller.get_current_ml.return_value.modules = {}
    controller.get_current_ml.return_value.waveforms = {}
    controller.arb_waveforms.list_data_keys.return_value = []
    controller.list_device_names.return_value = []
    return controller


@contextmanager
def _attached_form(
    schema: CfgSchema,
    ctrl: MagicMock,
    decoration_provider: FieldDecorationProvider | None = None,
) -> Generator[tuple[CfgFormWidget, CfgDraft]]:
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget(decoration_provider=decoration_provider)
    try:
        form.attach(draft)
        yield form, draft
    finally:
        form.detach()
        form.close()
        draft.close()
        form.deleteLater()


def _simple_schema() -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(
            fields={"reps": ScalarSpec(label="Reps", type=int, required=True)}
        ),
        value=CfgSectionValue(fields={"reps": DirectValue(10)}),
    )


def _complex_schema() -> CfgSchema:
    gauss_spec = CfgSectionSpec(
        label="Gaussian",
        fields={
            "sigma": ScalarSpec(label="Sigma", type=float, decimals=3),
            "length": ScalarSpec(label="Length", type=float, decimals=3),
        },
    )
    waveform_ref = ReferenceSpec(
        kind="waveform", label="Waveform", allowed=[gauss_spec], optional=True
    )
    pulse_spec = CfgSectionSpec(
        label="Pulse",
        fields={
            "freq": ScalarSpec(label="Freq", type=float, decimals=2),
            "waveform": waveform_ref,
            "sweep": SweepSpec(label="Sweep"),
            "csweep": CenteredSweepSpec(label="CSweep", decimals=2),
            "flag": ScalarSpec(label="Flag", type=bool),
            "choice": ScalarSpec(label="Mode", type=str, choices=["a", "b", "c"]),
            "eval": ScalarSpec(label="Eval", type=float),
        },
    )
    root_spec = CfgSectionSpec(
        label="Root",
        fields={
            "reps": ScalarSpec(label="Reps", type=int),
            "eval_direct": ScalarSpec(label="EvalDirect", type=float),
            "nested": pulse_spec,
            "ref": ReferenceSpec(kind="module", label="Ref", allowed=[pulse_spec]),
        },
    )
    gauss_val = CfgSectionValue(
        fields={
            "sigma": DirectValue(0.03),
            "length": EvalValue("2*sigma", resolved=0.06),
        }
    )
    pulse_val = CfgSectionValue(
        fields={
            "freq": EvalValue("r_f", resolved=6000.0),
            "waveform": ReferenceValue(chosen_key="<Custom:Gaussian>", value=gauss_val),
            "sweep": SweepValue(start=0.0, stop=1.0, expts=11),
            "csweep": CenteredSweepValue(center=0.5, span=1.0, expts=11),
            "flag": DirectValue(True),
            "choice": DirectValue("b"),
            "eval": DirectValue(3.14),
        }
    )
    return CfgSchema(
        spec=root_spec,
        value=CfgSectionValue(
            fields={
                "reps": DirectValue(42),
                "eval_direct": DirectValue(1.23),
                "nested": pulse_val,
                "ref": ReferenceValue(chosen_key="<Custom:Pulse>", value=pulse_val),
            }
        ),
    )


def test_tree_renders_and_edits_same_observable_behavior(qapp, ctrl):
    ctrl.get_current_md.return_value.r_f = 6000.0
    with _attached_form(_complex_schema(), ctrl) as (form, draft):
        values = form.read_values()
        assert values.fields["reps"] == DirectValue(42)
        nested = values.fields["nested"]
        assert isinstance(nested, CfgSectionValue)
        assert nested.fields["freq"] == EvalValue("r_f", resolved=6000.0)
        assert nested.fields["sweep"] == SweepValue(start=0.0, stop=1.0, expts=11)

        draft.set_target("reps", 999)
        draft.set_target("nested.choice", "c")
        draft.set_target("nested.flag", False)
        nested_field = draft.root.fields["nested"]
        assert isinstance(nested_field, SectionField)
        sweep = nested_field.fields["sweep"]
        centered = nested_field.fields["csweep"]
        assert isinstance(sweep, SweepField)
        assert isinstance(centered, CenteredSweepField)
        sweep.update_expts(5)
        centered.update_span(2.0)
        qapp.processEvents()

        values = form.read_values()
        assert values.fields["reps"] == DirectValue(999)
        nested = values.fields["nested"]
        assert isinstance(nested, CfgSectionValue)
        assert nested.fields["choice"] == DirectValue("c")
        assert nested.fields["flag"] == DirectValue(False)
        assert nested.fields["sweep"] == SweepValue(start=0.0, stop=1.0, expts=5)
        assert nested.fields["csweep"] == CenteredSweepValue(
            center=0.5, span=2.0, expts=11
        )
        reference = values.fields["ref"]
        assert isinstance(reference, ReferenceValue)
        assert reference.chosen_key == "<Custom:Pulse>"


def _assert_balanced_row(
    qapp: QApplication, left_label: QLabel, right_label: QLabel
) -> QWidget:
    left = left_label.parentWidget()
    right = right_label.parentWidget()
    assert left is not None and right is not None
    row = left.parentWidget()
    assert row is not None and row is right.parentWidget()
    row.resize(801, row.sizeHint().height())
    qapp.processEvents()
    assert abs(left.width() - right.width()) <= 1
    for cell in (left, right):
        layout = cell.layout()
        assert layout is not None
        editor_item = layout.itemAt(1)
        assert editor_item is not None
        editor = editor_item.widget()
        assert editor is not None
        assert editor.width() >= 20
    return row


def _assert_sweep_layout(
    qapp: QApplication, widget: QWidget, first_row_labels: tuple[str, str]
) -> None:
    labels = {label.text(): label for label in widget.findChildren(QLabel)}
    assert set(labels) == {*first_row_labels, "points", "step"}
    first_row = _assert_balanced_row(qapp, *(labels[text] for text in first_row_labels))
    second_row = _assert_balanced_row(qapp, labels["points"], labels["step"])
    assert first_row is not second_row
    assert first_row.y() < second_row.y()


def test_sweep_renderers_use_balanced_range_sampling_rows(qapp, ctrl):
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "ordinary": SweepSpec(label="Ordinary"),
                "centered": CenteredSweepSpec(label="Centered"),
            }
        ),
        value=CfgSectionValue(
            fields={
                "ordinary": SweepValue(start=0.0, stop=1.0, expts=11),
                "centered": CenteredSweepValue(center=0.5, span=1.0, expts=11),
            }
        ),
    )
    with _attached_form(schema, ctrl) as (form, _):
        form.resize(900, 500)
        form.show()
        qapp.processEvents()
        ordinary = form.findChild(SweepWidget)
        centered = form.findChild(CenteredSweepWidget)
        assert ordinary is not None and centered is not None
        _assert_sweep_layout(qapp, ordinary, ("start", "stop"))
        _assert_sweep_layout(qapp, centered, ("center", "span"))


def test_tree_whole_row_folding_is_view_only(qapp, ctrl):
    with _attached_form(_complex_schema(), ctrl) as (form, _):
        tree = tree_widget(form)
        root = tree.topLevelItem(0)
        assert root is not None and root.childCount() > 0
        before = form.read_values()
        expanded = root.isExpanded()
        click_row(qapp, form, root)
        assert root.isExpanded() != expanded
        click_row(qapp, form, root, column=1)
        assert root.isExpanded() == expanded
        assert form.read_values() == before
        leaf = tree_item(form, "reps")
        assert leaf.childCount() == 0
        leaf_expanded = leaf.isExpanded()
        click_row(qapp, form, leaf)
        assert leaf.isExpanded() == leaf_expanded


def _reference_schema() -> CfgSchema:
    gauss = CfgSectionSpec(
        label="Gauss", fields={"sigma": ScalarSpec(label="Sigma", type=float)}
    )
    return CfgSchema(
        spec=CfgSectionSpec(
            label="Root",
            fields={
                "ref": ReferenceSpec(
                    kind="waveform", label="Waveform", allowed=[gauss]
                ),
                "other": ScalarSpec(label="Other", type=int),
            },
        ),
        value=CfgSectionValue(
            fields={
                "ref": ReferenceValue(
                    chosen_key="<Custom:Gauss>",
                    value=CfgSectionValue(fields={"sigma": DirectValue(0.1)}),
                ),
                "other": DirectValue(1),
            }
        ),
    )


def test_tree_reference_shape_elision(qapp, ctrl):
    with _attached_form(_reference_schema(), ctrl) as (form, _):
        reference = tree_item(form, "ref")
        sigma = tree_item(form, "ref.sigma")
        assert reference.childCount() == 1
        assert reference.child(0) is sigma
        assert sigma.text(0) == "Sigma"
        assert tree_widget(form).itemWidget(sigma, 1) is not None


def test_tree_editing_lock_disables_editors(qapp, ctrl):
    with _attached_form(_simple_schema(), ctrl) as (form, _):
        tree = tree_widget(form)
        editor = tree.itemWidget(tree_item(form, "reps"), 1)
        assert editor is not None
        assert tree.isEnabled() and editor.isEnabled()
        form.set_editing_enabled(False)
        qapp.processEvents()
        assert not tree.isEnabled() and not editor.isEnabled()
        form.set_editing_enabled(True)
        assert tree.isEnabled() and editor.isEnabled()


def test_tree_detach_attach_preserves_draft(qapp, ctrl):
    with _attached_form(_simple_schema(), ctrl) as (form, draft):
        validity: list[bool] = []
        snapshots: list[CfgSchema] = []
        form.validity_changed.connect(validity.append)
        form.schema_changed.connect(snapshots.append)
        draft.set_target("reps", 11)
        qapp.processEvents()
        assert len(snapshots) == 1
        assert snapshots[0].value.fields["reps"] == DirectValue(11)

        other = MeasureCfgBindings(ctrl).new_draft(_simple_schema())
        try:
            form.detach()
            form.attach(other)
            qapp.processEvents()
            validity.clear()
            snapshots.clear()
            draft.set_target("reps", None)
            qapp.processEvents()
            draft.set_target("reps", 17)
            qapp.processEvents()
            assert validity == []
            assert snapshots == []
            assert form.read_values().fields["reps"] == DirectValue(10)

            form.detach()
            form.attach(draft)
            assert form.read_values().fields["reps"] == DirectValue(17)
            snapshots.clear()
            draft.set_target("reps", 18)
            qapp.processEvents()
            assert len(snapshots) == 1
            assert snapshots[0].value.fields["reps"] == DirectValue(18)
            draft.set_target("reps", None)
            qapp.processEvents()
            assert validity[-1] is False
        finally:
            form.detach()
            other.close()


def test_tree_validation_propagation(qapp, ctrl):
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"v": ScalarSpec(label="V", type=int, required=True)}
        ),
        value=CfgSectionValue(fields={"v": DirectValue(1)}),
    )
    with _attached_form(schema, ctrl) as (form, draft):
        events: list[bool] = []
        form.validity_changed.connect(events.append)
        form.attach(draft)
        qapp.processEvents()
        assert events == [True]
        assert form.is_valid()
        draft.set_target("v", None)
        qapp.processEvents()
        assert events[-1] is False
        assert not form.is_valid()
        assert form.first_invalid_reason() is not None
        draft.set_target("v", 2)
        qapp.processEvents()
        assert events[-1] is True
        assert form.is_valid()


def test_tree_section_local_refresh_choice(qapp, ctrl):
    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half": ScalarSpec(label="Half", type=float),
        "manual": ScalarSpec(label="Manual", type=float),
    }
    choice = ChoiceSectionSpec(
        label="Choice",
        fields=fields,
        bindings=(
            ChoiceBinding(
                "mode",
                {
                    "auto": CfgSectionSpec(fields={"half": fields["half"]}),
                    "fixed": CfgSectionSpec(fields={"manual": fields["manual"]}),
                },
            ),
        ),
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"choice": choice, "stable": ScalarSpec(label="Stable", type=float)}
        ),
        value=CfgSectionValue(
            fields={
                "choice": CfgSectionValue(
                    fields={
                        "mode": DirectValue("auto"),
                        "half": DirectValue(1.0),
                        "manual": DirectValue(2.0),
                    }
                ),
                "stable": DirectValue(3.0),
            }
        ),
    )
    with _attached_form(schema, ctrl) as (form, draft):
        assert "choice.half" in form.decoration_paths()
        assert "choice.manual" not in form.decoration_paths()
        draft.set_target("choice.mode", "fixed")
        paths = form.decoration_paths()
        assert "choice.half" not in paths
        assert "choice.manual" in paths
        assert "stable" in paths


def test_tree_shares_same_draft_binding_ref_identity(qapp, ctrl):
    with _attached_form(_reference_schema(), ctrl) as (first, draft):
        second = CfgFormWidget()
        try:
            first.detach()
            second.attach(draft)
            assert draft.resolve_target("ref.sigma") is not None
            draft.set_target("ref.sigma", 0.99)
            qapp.processEvents()
            expected = ReferenceValue(
                chosen_key="<Custom:Gauss>",
                value=CfgSectionValue(fields={"sigma": DirectValue(0.99)}),
                resolved_label="Gauss",
            )
            assert second.read_values().fields["ref"] == expected
            second.detach()
            first.attach(draft)
            assert first.read_values().fields["ref"] == expected
        finally:
            second.detach()
            second.close()
            second.deleteLater()


@dataclass(frozen=True)
class _DisabledPaths:
    paths: tuple[str, ...]

    def decoration_for(
        self, path: str, spec: object, value: object
    ) -> FieldDecorationPatch | None:
        del spec, value
        if path in self.paths:
            return FieldDecorationPatch(enabled=False, badge="disabled")
        return None


def _enabled_states(form: CfgFormWidget, paths: tuple[str, ...]) -> dict[str, bool]:
    tree = tree_widget(form)
    states: dict[str, bool] = {}
    for path in paths:
        item = tree_item(form, path)
        editor = tree.itemWidget(item, 1)
        states[path] = not item.isDisabled() and (editor is None or editor.isEnabled())
    return states


def _optional_outer_schema(shape: CfgSectionSpec, value: CfgSectionValue) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "outer": ReferenceSpec(
                    kind="module", label="Outer", allowed=[shape], optional=True
                )
            }
        ),
        value=CfgSectionValue(
            fields={
                "outer": ReferenceValue(chosen_key="<Custom:OuterShape>", value=value)
            }
        ),
    )


def test_tree_outer_reenable_preserves_nested_reference_and_decoration(qapp, ctrl):
    inner = CfgSectionSpec(
        label="InnerShape",
        fields={"inner_leaf": ScalarSpec(label="InnerLeaf", type=int)},
    )
    shape = CfgSectionSpec(
        label="OuterShape",
        fields={
            "inner_ref": ReferenceSpec(
                kind="module", label="Inner", allowed=[inner], optional=True
            ),
            "deco_leaf": ScalarSpec(label="Deco", type=int),
            "normal_leaf": ScalarSpec(label="Normal", type=int),
        },
    )
    schema = _optional_outer_schema(
        shape,
        CfgSectionValue(
            fields={"deco_leaf": DirectValue(10), "normal_leaf": DirectValue(20)}
        ),
    )
    enabled = {
        "outer.inner_ref": True,
        "outer.inner_ref.inner_leaf": False,
        "outer.deco_leaf": False,
        "outer.normal_leaf": True,
    }
    with _attached_form(schema, ctrl, _DisabledPaths(("outer.deco_leaf",))) as (
        form,
        draft,
    ):
        outer = draft.root.fields["outer"]
        assert isinstance(outer, ReferenceField) and outer.is_enabled
        assert outer.sub_field is not None
        inner_field = outer.sub_field.fields["inner_ref"]
        assert isinstance(inner_field, ReferenceField) and not inner_field.is_enabled
        assert _enabled_states(form, tuple(enabled)) == enabled
        outer.set_enabled(False)
        qapp.processEvents()
        assert not outer.is_enabled
        assert _enabled_states(form, tuple(enabled)) == dict.fromkeys(enabled, False)
        outer.set_enabled(True)
        qapp.processEvents()
        assert outer.is_enabled and not inner_field.is_enabled
        assert _enabled_states(form, tuple(enabled)) == enabled
        assert form.decoration_for_path("outer.deco_leaf").enabled is False
        assert form.decoration_for_path("outer.normal_leaf").enabled is True


def test_tree_outer_reenable_preserves_decoration_disabled_containers(qapp, ctrl):
    section = CfgSectionSpec(
        label="InnerSection", fields={"sec_leaf": ScalarSpec(label="SecLeaf", type=int)}
    )
    reference_shape = CfgSectionSpec(
        label="InnerRefShape",
        fields={"ref_leaf": ScalarSpec(label="RefLeaf", type=int)},
    )
    shape = CfgSectionSpec(
        label="OuterShape",
        fields={
            "inner_section": section,
            "inner_ref": ReferenceSpec(
                kind="module", label="InnerRef", allowed=[reference_shape]
            ),
            "normal_leaf": ScalarSpec(label="Normal", type=int),
        },
    )
    schema = _optional_outer_schema(
        shape,
        CfgSectionValue(
            fields={
                "inner_section": CfgSectionValue(fields={"sec_leaf": DirectValue(7)}),
                "inner_ref": ReferenceValue(
                    chosen_key="<Custom:InnerRefShape>",
                    value=CfgSectionValue(fields={"ref_leaf": DirectValue(5)}),
                ),
                "normal_leaf": DirectValue(1),
            }
        ),
    )
    disabled = ("outer.inner_section", "outer.inner_ref")
    enabled = {
        "outer.inner_section": False,
        "outer.inner_section.sec_leaf": False,
        "outer.inner_ref": False,
        "outer.inner_ref.ref_leaf": False,
        "outer.normal_leaf": True,
    }
    with _attached_form(schema, ctrl, _DisabledPaths(disabled)) as (form, draft):
        outer = draft.root.fields["outer"]
        assert isinstance(outer, ReferenceField)
        assert _enabled_states(form, tuple(enabled)) == enabled
        assert form.decoration_for_path("outer.inner_section.sec_leaf").enabled is True
        assert form.decoration_for_path("outer.inner_ref.ref_leaf").enabled is True
        for outer_enabled in (False, True):
            outer.set_enabled(outer_enabled)
            qapp.processEvents()
            expected = enabled if outer_enabled else dict.fromkeys(enabled, False)
            assert _enabled_states(form, tuple(enabled)) == expected
            assert form.decoration_for_path("outer.inner_section").enabled is False
            assert form.decoration_for_path("outer.inner_ref").enabled is False
