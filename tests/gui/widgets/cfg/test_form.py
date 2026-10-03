"""Tests for shared CfgFormWidget populate and read-values behavior."""

from __future__ import annotations

from typing import Any, Literal, cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QComboBox, QLabel, QLineEdit
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
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
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgField,
    LiteralField,
    ReferenceField,
    ScalarField,
    SectionField,
    SweepField,
)
from zcu_tools.gui.widgets.cfg import (
    CfgFormWidget,
    FieldRenderContext,
    FieldRenderer,
    FieldRendererRegistry,
    FrozenFieldRendererRegistry,
    default_cfg_renderers,
)
from zcu_tools.gui.widgets.cfg.fields import CenteredSweepWidget, ReferenceWidget
from zcu_tools.gui.widgets.cfg.registry import FieldWidgetProtocol
from zcu_tools.resources.context import ModuleLibrary

from tests.gui.widgets.cfg._form_support import (
    attach_draft,
    scalar_field,
    section_schema,
)
from tests.gui.widgets.cfg._tree_support import tree_item, tree_widget

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_RENDERED_FIELD_TYPES = (
    LiteralField,
    ScalarField,
    SweepField,
    CenteredSweepField,
    ReferenceField,
)


def _registry_with_factories(
    overrides: dict[type[CfgField], FieldRenderer],
) -> FrozenFieldRendererRegistry:
    defaults = default_cfg_renderers()
    builder = FieldRendererRegistry()
    for field_type in _RENDERED_FIELD_TYPES:
        builder.register(
            field_type,
            overrides.get(field_type, defaults.resolve(field_type)),
        )
    # SectionField is structural (sole tree) and has no default registry entry.
    # Tests that need a SectionField factory provide it explicitly via overrides.
    if SectionField in overrides:
        builder.register(SectionField, overrides[SectionField])
    return builder.freeze()


def _make_ctx():
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    return SessionEnv(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None)


# ---------------------------------------------------------------------------
# schema_to_dict — SweepValue
# ---------------------------------------------------------------------------


def test_sweep_value_uses_expts_as_canonical():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=1.0, stop=2.0, expts=5, step=999.0)},
    )
    result = schema_to_raw_dict(schema, None, ml)
    assert isinstance(result["f"], SweepCfg)
    assert result["f"].expts == 5
    assert result["f"].step == pytest.approx(0.25)


def test_sweep_value_step_mode():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    result = schema_to_raw_dict(schema, None, ml)
    sweep = result["f"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.step == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# make_scalar_widget / read_scalar_widget
# ---------------------------------------------------------------------------


def test_scalar_int_widget_round_trip(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="X", type=int)
    w = make_scalar_widget(spec, 42)
    assert read_scalar_widget(w, spec) == 42


def test_scalar_float_widget_round_trip(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Pi", type=float)
    w = make_scalar_widget(spec, 3.14)
    assert read_scalar_widget(w, spec) == pytest.approx(3.14)


def test_scalar_bool_widget_round_trip(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Flag", type=bool)
    w = make_scalar_widget(spec, True)
    assert read_scalar_widget(w, spec) is True


def test_scalar_choices_widget_round_trip(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Model", type=str, choices=["hm", "t", "auto"])
    w = make_scalar_widget(spec, "hm")
    assert read_scalar_widget(w, spec) == "hm"


def test_dynamic_arb_waveform_data_choices(qapp, ctrl):
    ctrl.arb_waveforms.list_data_keys.return_value = ["asset_a", "asset_b"]
    schema = section_schema(
        {
            "data": ScalarSpec(
                label="Data key",
                type=str,
                required=True,
                choices_source="arb_waveforms",
            )
        },
        {"data": DirectValue(None)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)

        combo = form.findChild(QComboBox)
        assert combo is not None
        assert [combo.itemText(i) for i in range(combo.count())] == ["asset_a", "asset_b"]
        assert combo.currentIndex() == -1
        assert not form.is_valid()

        combo.setCurrentIndex(1)

        value = form.read_values().fields["data"]
        assert isinstance(value, DirectValue) and value.value == "asset_b"
        assert form.is_valid()
    finally:
        form.detach()
        form.close()
        draft.close()


def test_arb_waveform_data_choice_allows_empty_initial_value(qapp, ctrl):
    ctrl.arb_waveforms.list_data_keys.return_value = ["asset_a"]
    schema = section_schema(
        {
            "data": ScalarSpec(
                label="Data key",
                type=str,
                choices_source="arb_waveforms",
            )
        },
        {"data": DirectValue("")},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)

        combo = form.findChild(QComboBox)
        assert combo is not None
        assert [combo.itemText(i) for i in range(combo.count())] == ["", "asset_a"]
        assert combo.currentIndex() == 0
        assert form.is_valid()

        value = form.read_values().fields["data"]
        assert isinstance(value, DirectValue) and value.value == ""
    finally:
        form.detach()
        form.close()
        draft.close()


def test_dynamic_choice_renders_inactive_current_value_but_remains_invalid(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctrl.arb_waveforms.list_data_keys.return_value = []
    schema = section_schema(
        {
            "data": ScalarSpec(
                label="Data key",
                type=str,
                required=True,
                choices_source="arb_waveforms",
            )
        },
        {"data": DirectValue("retired_asset")},
    )
    form = CfgFormWidget()
    attach_draft(form, schema, ctrl)

    combo = form.findChild(QComboBox)
    assert combo is not None
    assert [combo.itemText(index) for index in range(combo.count())] == [
        "retired_asset"
    ]
    assert combo.currentText() == "retired_asset"
    assert not form.is_valid()


def test_scalar_editable_false_widget_disabled(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget

    spec = ScalarSpec(label="RO", type=float, editable=False)
    w = make_scalar_widget(spec, 1.0)
    assert not w.isEnabled()


def test_optional_scalar_widget_is_line_edit_empty_for_none(qapp):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Mixer freq", type=float, optional=True)
    # None → an empty QLineEdit (spinbox cannot show "unset"); reads back as None.
    w = make_scalar_widget(spec, "")
    assert isinstance(w, QLineEdit)
    assert w.text() == ""
    assert read_scalar_widget(w, spec) is None


def test_optional_scalar_widget_round_trips_value(qapp):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Mixer freq", type=float, optional=True)
    w = make_scalar_widget(spec, 5000.0)
    assert isinstance(w, QLineEdit)
    assert read_scalar_widget(w, spec) == pytest.approx(5000.0)
    # Clearing the field reads back as None (unset).
    w.setText("")
    assert read_scalar_widget(w, spec) is None


def test_form_propagates_renderer_registry_through_reference_subtree(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit, QWidget
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    contexts: list[FieldRenderContext] = []
    defaults = default_cfg_renderers()

    def recording_factory(
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        contexts.append(context)
        widget = defaults.resolve(field)(field, context)
        assert isinstance(widget, QWidget)
        widget.setObjectName(f"custom-{context.path}")
        return widget

    inner_spec = CfgSectionSpec(
        label="Inner",
        fields={"value": ScalarSpec(label="Value", type=int)},
    )
    schema = section_schema(
        {"reference": ReferenceSpec(kind="module", allowed=[inner_spec])},
        {
            "reference": ReferenceValue(
                chosen_key="<Custom:Inner>",
                value=CfgSectionValue(fields={"value": DirectValue(1)}),
            )
        },
    )
    registry = _registry_with_factories(
        {ReferenceField: recording_factory, ScalarField: recording_factory}
    )
    form = CfgFormWidget(renderers=registry)

    attach_draft(form, schema, ctrl)

    assert sorted(context.path for context in contexts) == [
        "reference",
        "reference.value",
    ]
    assert all(context.registry is registry for context in contexts)
    assert form.findChild(QWidget, "custom-reference") is not None
    leaf = form.findChild(QWidget, "custom-reference.value")
    assert leaf is not None
    editor = leaf.findChild(QLineEdit)
    assert editor is not None
    assert editor.text() == "1"


def test_scalar_widget_minimum_width_reduced(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget

    spec = ScalarSpec(label="Name", type=str)
    w = make_scalar_widget(spec, "demo")
    assert w.minimumWidth() == 20


def test_scalar_widget_eval_mode_shows_resolved_ghost(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    field = scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("r_f"),
    )

    w = ScalarWidget(field)
    assert w._ghost is not None
    assert w._ghost.text() == "= 6000.0"


def test_scalar_widget_eval_mode_marks_unresolved_red(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    ctrl.get_current_md.return_value = MetaDict()
    field = scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("missing"),
    )

    w = ScalarWidget(field)
    assert w._ghost is not None
    assert w._ghost.text() == "= ?"
    assert "red" in w._ghost.styleSheet()


def test_measure_cfg_form_value_source_resolves_on_space_in_eval_input(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.app.measure.ui.cfg_binding import (
        make_value_source_input_enhancer,
    )
    from zcu_tools.gui.session.value_lookup import ValueInfo
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    ctrl.read_value_source.return_value = (
        ValueInfo("device.flux.value", float, "device:flux"),
        0.125,
    )
    schema = section_schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue("r_f")},
    )
    form = CfgFormWidget(text_input_enhancer=make_value_source_input_enhancer(ctrl))
    root = attach_draft(form, schema, ctrl)
    scalar_widget = form.findChild(ScalarWidget)
    edit = form.findChild(QLineEdit)
    assert scalar_widget is not None
    assert edit is not None

    edit.setText("@{device.flux.value} ")
    edit.setCursorPosition(len(edit.text()))
    enhancer = scalar_widget._input_enhancement
    assert enhancer is not None
    cast(Any, enhancer)._on_text_edited(edit.text())

    value = cast(ScalarField, root.fields["freq"]).get_value()
    assert isinstance(value, EvalValue)
    assert value.expr == "0.125"
    assert edit.text() == "0.125"
    ctrl.read_value_source.assert_called_once_with("device.flux.value")


def test_scalar_widget_eval_menu_extends_standard_line_edit_menu(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    field = scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("r_f"),
    )
    w = ScalarWidget(field)
    edit = w.findChild(QLineEdit)
    assert edit is not None

    menu, mode_action = w._build_context_menu(edit)
    action_texts = [action.text() for action in menu.actions()]

    assert mode_action is not None
    assert "Use direct value" in action_texts
    assert len(action_texts) > 1


def test_scalar_widget_unresolved_eval_can_switch_back_to_direct(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    ctrl.get_current_md.return_value = MetaDict()
    field = scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("missing"),
    )
    w = ScalarWidget(field)

    field.set_value(None)

    value = field.get_value()
    assert isinstance(value, DirectValue)
    # unset scalar is value=None (ADR-0010) — no placeholder default
    assert value.value is None
    entry = w.findChild(QLineEdit)
    assert entry is not None
    assert entry.text() == ""
    assert not field.is_valid()


# ---------------------------------------------------------------------------
# CfgFormWidget — populate and read_values / read_schema
# ---------------------------------------------------------------------------


def test_read_values_before_populate_raises(qapp):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_values()


def test_read_schema_before_populate_raises(qapp):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_schema()


def test_populate_scalar_fields_round_trip(qapp, ctrl):
    schema = section_schema(
        {
            "reps": ScalarSpec(label="Reps", type=int),
            "freq": ScalarSpec(label="Freq", type=float),
        },
        {
            "reps": DirectValue(100),
            "freq": DirectValue(6.0),
        },
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        values = form.read_values()
        reps = values.fields["reps"]
        freq = values.fields["freq"]
        assert isinstance(reps, DirectValue) and reps.value == 100
        assert isinstance(freq, DirectValue) and freq.value == pytest.approx(6.0)
    finally:
        form.detach()
        form.close()
        draft.close()


@pytest.mark.parametrize("failure", ["invalid_widget", "exception"])
def test_attach_failure_does_not_observe_failed_draft(
    qapp, ctrl, failure: Literal["invalid_widget", "exception"]
):
    from qtpy.QtWidgets import QWidget
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"value": ScalarSpec(label="Value", type=int, required=True)},
        {"value": DirectValue(1)},
    )
    failed_draft = MeasureCfgBindings(ctrl).new_draft(schema)
    active_draft = MeasureCfgBindings(ctrl).new_draft(schema)
    fail_rendering = True
    default_renderer = default_cfg_renderers().resolve(ScalarField)

    def recovering_factory(
        field: CfgField, context: FieldRenderContext
    ) -> FieldWidgetProtocol:
        if fail_rendering:
            if failure == "invalid_widget":
                # Exercise runtime protocol rejection at the renderer boundary.
                return cast(FieldWidgetProtocol, QWidget())
            raise RuntimeError("factory exploded")
        return default_renderer(field, context)

    error_type = TypeError if failure == "invalid_widget" else RuntimeError
    message = (
        "expected FieldWidgetProtocol"
        if failure == "invalid_widget"
        else "factory exploded"
    )
    form = CfgFormWidget(
        renderers=_registry_with_factories({ScalarField: recovering_factory}),
    )
    validity: list[bool] = []
    schemas: list[CfgSchema] = []
    form.validity_changed.connect(validity.append)
    form.schema_changed.connect(schemas.append)
    try:
        with pytest.raises(error_type, match=message):
            form.attach(failed_draft)
        with pytest.raises(RuntimeError, match=r"attach\(\) must be called"):
            form.read_values()
        assert form.decoration_paths() == ()
        assert validity == []

        fail_rendering = False
        form.attach(active_draft)
        assert validity == [True]

        failed_draft.set_target("value", None)
        qapp.processEvents()
        assert validity == [True]
        assert schemas == []
        assert form.read_values().fields["value"] == DirectValue(1)

        active_draft.set_target("value", None)
        assert validity == [True, False]
        qapp.processEvents()
        assert len(schemas) == 1
        assert schemas[0].value.fields["value"] == DirectValue(None)
    finally:
        form.detach()
        failed_draft.close()
        active_draft.close()


def test_detach_and_reattach_validity_subscription_emits_once(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"value": ScalarSpec(label="Value", type=int, required=True)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    validity: list[bool] = []
    form.validity_changed.connect(validity.append)

    try:
        form.attach(draft)
        assert validity == [True]

        form.detach()
        draft.set_target("value", None)
        draft.set_target("value", 2)
        assert validity == [True]

        form.attach(draft)
        assert validity == [True, True]

        draft.set_target("value", None)
        assert validity == [True, True, False]
    finally:
        form.detach()
        draft.close()


def test_set_editing_enabled_keeps_scroll_area_enabled(qapp, ctrl):
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QLineEdit, QScrollArea
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    w = CfgFormWidget()
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    try:
        scroll = w.findChild(QScrollArea)
        assert scroll is not None
        assert w.isEnabled()
        assert scroll.isEnabled()

        tree = tree_widget(w)
        spin = tree.findChild(QLineEdit)
        assert spin is not None
        assert spin.isEnabled()

        w.set_editing_enabled(False)

        assert w.isEnabled()
        assert scroll.isEnabled()
        assert not tree.isEnabled()
        assert not spin.isEnabled()

        w.detach()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        w.attach(draft)

        assert w.isEnabled()
        assert scroll.isEnabled()
        reattached_tree = tree_widget(w)
        assert not reattached_tree.isEnabled()
        reattached_spin = reattached_tree.findChild(QLineEdit)
        assert reattached_spin is not None
        assert not reattached_spin.isEnabled()

        w.set_editing_enabled(True)

        assert reattached_tree.isEnabled()
        assert reattached_spin.isEnabled()
    finally:
        w.detach()
        draft.close()


def test_cfg_form_reflects_model_external_refresh(qapp, ctrl):
    """The widget reflects expression refreshes from its attached draft."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.resources.context import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    schema = section_schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue("r_f")},
    )
    w = CfgFormWidget()
    emitted = []
    w.schema_changed.connect(emitted.append)
    model = attach_draft(w, schema, ctrl)

    md.r_f = 6100.0
    model.refresh_expressions()
    qapp.processEvents()

    val = w.read_values().fields["freq"]
    assert isinstance(val, EvalValue)
    assert val.resolved == 6100.0
    assert emitted


def test_same_tick_edits_materialize_schema_once_at_form_boundary(
    qapp, ctrl, monkeypatch: pytest.MonkeyPatch
):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"nested": CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)})},
        {"nested": CfgSectionValue({"reps": DirectValue(10)})},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    form.attach(draft)
    emitted: list[CfgSchema] = []
    form.schema_changed.connect(emitted.append)
    snapshot_count = 0
    original_snapshot = draft.snapshot

    def count_snapshot() -> CfgSchema:
        nonlocal snapshot_count
        snapshot_count += 1
        return original_snapshot()

    monkeypatch.setattr(draft, "snapshot", count_snapshot)
    try:
        draft.set_target("nested.reps", 11)
        draft.set_target("nested.reps", 12)
        draft.set_target("nested.reps", 13)

        assert snapshot_count == 0
        assert emitted == []

        qapp.processEvents()

        assert snapshot_count == 1
        assert len(emitted) == 1
        nested_value = emitted[0].value.fields["nested"]
        assert isinstance(nested_value, CfgSectionValue)
        assert nested_value.fields["reps"] == DirectValue(13)
    finally:
        form.detach()
        draft.close()


def test_validity_feedback_stays_synchronous_while_schema_is_coalesced(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"value": ScalarSpec(label="Value", type=int, required=True)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    validity: list[bool] = []
    schemas: list[CfgSchema] = []
    form.validity_changed.connect(validity.append)
    form.schema_changed.connect(schemas.append)
    form.attach(draft)

    try:
        draft.set_target("value", None)
        draft.set_target("value", 2)
        draft.set_target("value", None)

        assert validity == [True, False, True, False]
        assert schemas == []

        qapp.processEvents()

        assert len(schemas) == 1
        assert schemas[0].value.fields["value"] == DirectValue(None)
    finally:
        form.detach()
        draft.close()


def test_detach_drops_pending_schema_and_reattach_can_schedule(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"value": ScalarSpec(label="Value", type=int)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    schemas: list[CfgSchema] = []
    form.schema_changed.connect(schemas.append)
    form.attach(draft)

    try:
        draft.set_target("value", 2)
        form.detach()
        qapp.processEvents()

        assert schemas == []

        form.attach(draft)
        draft.set_target("value", 3)
        qapp.processEvents()

        assert len(schemas) == 1
        assert schemas[0].value.fields["value"] == DirectValue(3)
    finally:
        form.detach()
        draft.close()


def test_close_drops_pending_schema_and_reattach_can_schedule(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"value": ScalarSpec(label="Value", type=int)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    schemas: list[CfgSchema] = []
    form.schema_changed.connect(schemas.append)
    form.attach(draft)
    form.show()

    try:
        draft.set_target("value", 2)
        form.close()
        qapp.processEvents()

        assert schemas == []

        form.attach(draft)
        draft.set_target("value", 3)
        qapp.processEvents()

        assert len(schemas) == 1
        assert schemas[0].value.fields["value"] == DirectValue(3)
    finally:
        form.detach()
        draft.close()


def test_cfg_form_does_not_subscribe_bus(qapp, ctrl):
    """Attach/detach never registers an EventBus subscription."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": DirectValue(6000.0)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    attach_draft(w, schema, ctrl)  # re-attach swaps models cleanly

    bus = ctrl.get_bus.return_value
    assert bus._subs == {} or all(not subs for subs in bus._subs.values())


def test_read_schema_returns_cfg_schema(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(10)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    out = w.read_schema()
    assert isinstance(out, CfgSchema)
    assert out.spec is schema.spec


def test_read_values_does_not_mutate_original(qapp, ctrl):
    schema = section_schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)

        entry = form.findChild(QLineEdit)
        assert entry is not None
        entry.setText("999")

        reps = form.read_values().fields["reps"]
        original_reps = schema.value.fields["reps"]
        assert isinstance(reps, DirectValue) and reps.value == 999
        assert isinstance(original_reps, DirectValue) and original_reps.value == 100
    finally:
        form.detach()
        form.close()
        draft.close()


def test_populate_sweep_field_round_trip(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=5.8, stop=6.2, expts=201)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.start == pytest.approx(5.8)
    assert sv.stop == pytest.approx(6.2)
    assert sv.expts == 201
    assert sv.step == pytest.approx(0.002)


def test_populate_centered_sweep_field_round_trip(qapp, ctrl):
    schema = section_schema(
        {
            "f": CenteredSweepSpec(
                label="Freq",
                center_editable=False,
                center_badge="generated",
                center_tooltip="Generated center",
            )
        },
        {"f": CenteredSweepValue(center=0.0, span=100.0, expts=201)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        sweep_widget = form.findChild(CenteredSweepWidget)
        assert sweep_widget is not None

        span_input = sweep_widget.findChild(QLineEdit, "span")
        points_input = sweep_widget.findChild(QLineEdit, "expts")
        assert span_input is not None
        assert points_input is not None
        center_input = sweep_widget.findChild(QLineEdit)
        assert center_input is not None
        assert not center_input.isEnabled()
        labels = {label.text(): label for label in sweep_widget.findChildren(QLabel)}
        center_label = labels["center [generated]"]
        assert center_label.toolTip() == "Generated center"
        center_cell = center_label.parentWidget()
        span_cell = labels["span"].parentWidget()
        assert center_cell is not None
        assert span_cell is not None
        pair_row = center_cell.parentWidget()
        assert pair_row is span_cell.parentWidget()
        assert pair_row is not None
        pair_row.resize(801, pair_row.sizeHint().height())
        qapp.processEvents()
        assert abs(center_cell.width() - span_cell.width()) <= 1

        span_input.setText("120.0")
        points_input.setText("121")
        sv = form.read_values().fields["f"]
        assert isinstance(sv, CenteredSweepValue)
        assert sv.center == pytest.approx(0.0)
        assert sv.span == DirectValue(120.0, raw="120.0")
        assert sv.expts == DirectValue(121, raw="121")
        assert sv.step == pytest.approx(1.0)
        assert form.is_valid()

        span_input.setText("0.0")
        sv = form.read_values().fields["f"]
        assert isinstance(sv, CenteredSweepValue)
        assert isinstance(sv.span, DirectValue)
        assert sv.span.value is None
        assert sv.span.raw == "0.0"
        assert sv.span.error is not None
        assert span_input.text() == "0.0"
        assert not form.is_valid()
    finally:
        form.detach()
        form.close()
        draft.close()


def test_populate_sweep_field_step_preserved(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.1)


def test_sweep_widget_step_change_recomputes_expts_and_stop(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    entry = sweep_widget.findChild(QLineEdit, "step")
    assert entry is not None
    entry.setText("0.2")
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.expts == 6
    assert sv.stop == pytest.approx(1.0)
    assert sv.step == DirectValue(0.2, raw="0.2")


def test_sweep_widget_non_step_change_recomputes_step(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    entry = sweep_widget.findChild(QLineEdit, "expts")
    assert entry is not None
    entry.setText("5")
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.25)


def test_sweep_widget_start_supports_eval_mode(qapp, ctrl):
    from zcu_tools.gui.cfg import EvalValue
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    sweep_widget._field.start_field.set_value(
        EvalValue(expr="r_f - 1", resolved=5999.0)
    )
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert isinstance(sv.start, EvalValue)
    assert sv.start.expr == "r_f - 1"


def test_populate_nested_section_round_trip(qapp, ctrl):
    schema = section_schema(
        {
            "inner": CfgSectionSpec(
                fields={"gain": ScalarSpec(label="Gain", type=float)}
            )
        },
        {"inner": CfgSectionValue(fields={"gain": DirectValue(0.05)})},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        inner = form.read_values().fields["inner"]
        assert isinstance(inner, CfgSectionValue)
        gain = inner.fields["gain"]
        assert isinstance(gain, DirectValue) and gain.value == pytest.approx(0.05)
    finally:
        form.detach()
        form.close()
        draft.close()


def test_nested_sections_render_without_outer_duplicate_label(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {
            "inner": CfgSectionSpec(
                label="Inner",
                fields={"gain": ScalarSpec(label="Gain", type=float)},
            )
        },
        {"inner": CfgSectionValue(fields={"gain": DirectValue(0.05)})},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)

    # Sole tree: inner section is a QTreeWidgetItem header, not a QLabel with "<b>Inner</b>"
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    root = w._root_widget
    assert isinstance(root, TreeCfgWidget)
    # Find Inner header item
    found_inner = False
    found_gain = False
    stack = [root._tree.invisibleRootItem()]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        for i in range(cur.childCount()):
            child = cur.child(i)
            if child is None:
                continue
            txt = child.text(0)
            if txt == "Inner":
                found_inner = True
            if "Gain" in txt:
                found_gain = True
            stack.append(child)
    assert found_inner
    assert found_gain


def test_choice_section_renders_only_active_choice_fields(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half_width": ScalarSpec(label="Half width", type=float),
        "decay": ScalarSpec(label="Decay", type=float),
        "manual_value": ScalarSpec(label="Manual", type=float),
    }
    schema = section_schema(
        {
            "search": ChoiceSectionSpec(
                label="Search",
                fields=fields,
                bindings=(
                    ChoiceBinding(
                        "mode",
                        {
                            "auto": CfgSectionSpec(
                                fields={
                                    "half_width": fields["half_width"],
                                    "decay": fields["decay"],
                                }
                            ),
                            "fixed": CfgSectionSpec(
                                fields={"manual_value": fields["manual_value"]}
                            ),
                        },
                    ),
                ),
            )
        },
        {
            "search": CfgSectionValue(
                fields={
                    "mode": DirectValue("auto"),
                    "half_width": DirectValue(1.0),
                    "decay": DirectValue(3.0),
                    "manual_value": DirectValue(2.0),
                }
            )
        },
    )
    w = CfgFormWidget()
    model = attach_draft(w, schema, ctrl)

    paths = set(w.decoration_paths())
    assert "search.mode" in paths
    assert "search.half_width" in paths
    assert "search.decay" in paths
    assert "search.manual_value" not in paths

    search = model.fields["search"]
    assert isinstance(search, SectionField)
    search.fields["mode"].set_value(DirectValue("fixed"))

    paths = set(w.decoration_paths())
    assert "search.mode" in paths
    assert "search.half_width" not in paths
    assert "search.decay" not in paths
    assert "search.manual_value" in paths

    out = w.read_values().fields["search"]
    assert isinstance(out, CfgSectionValue)
    assert set(out.fields) == {"mode", "half_width", "decay", "manual_value"}


def test_choice_section_rebuilds_only_changed_section(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half_width": ScalarSpec(label="Half width", type=float),
        "manual_value": ScalarSpec(label="Manual", type=float),
    }
    schema = section_schema(
        {
            "search": ChoiceSectionSpec(
                label="Search",
                fields=fields,
                bindings=(
                    ChoiceBinding(
                        "mode",
                        {
                            "auto": CfgSectionSpec(
                                fields={"half_width": fields["half_width"]}
                            ),
                            "fixed": CfgSectionSpec(
                                fields={"manual_value": fields["manual_value"]}
                            ),
                        },
                    ),
                ),
            ),
            "stable": ScalarSpec(label="Stable", type=float),
        },
        {
            "search": CfgSectionValue(
                fields={
                    "mode": DirectValue("auto"),
                    "half_width": DirectValue(1.0),
                    "manual_value": DirectValue(2.0),
                }
            ),
            "stable": DirectValue(3.0),
        },
    )
    from qtpy.QtCore import Qt

    w = CfgFormWidget()
    model = attach_draft(w, schema, ctrl)
    root_widget = w._root_widget
    assert isinstance(root_widget, TreeCfgWidget)
    # Choice decoration paths should update section-locally and keep widget instance
    w.decoration_paths()
    assert "search.half_width" in w.decoration_paths()
    # Capture unrelated subtree widget identity before change
    stable_before = root_widget._leaf_path_to_widget["stable"]
    stable_item_before = root_widget._tree.findItems(  # type: ignore[attr-defined]
        "Stable",
        Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive,
        0,  # type: ignore[attr-defined]
    )
    assert stable_item_before
    search = model.fields["search"]
    assert isinstance(search, SectionField)
    # Capture search's half_width widget before (should be replaced)
    half_before = root_widget._leaf_path_to_widget.get("search.half_width")
    assert half_before is not None
    search.fields["mode"].set_value(DirectValue("fixed"))
    w.decoration_paths()

    assert w._root_widget is root_widget
    assert "search.half_width" not in w.decoration_paths()
    assert "search.manual_value" in w.decoration_paths()
    # Unrelated leaf "stable" must retain same widget/item (section-local)
    assert root_widget._leaf_path_to_widget["stable"] is stable_before
    # Changed section's old leaf should be gone, new leaf should be present and different
    assert "search.half_width" not in root_widget._leaf_path_to_widget
    manual_after = root_widget._leaf_path_to_widget.get("search.manual_value")
    assert manual_after is not None
    assert manual_after is not half_before


def test_choice_refresh_fallback_preserves_pending_schema_snapshot(
    qapp, ctrl, monkeypatch: pytest.MonkeyPatch
):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half_width": ScalarSpec(label="Half width", type=float),
        "manual_value": ScalarSpec(label="Manual", type=float),
    }
    schema = section_schema(
        {
            "search": ChoiceSectionSpec(
                fields=fields,
                bindings=(
                    ChoiceBinding(
                        "mode",
                        {
                            "auto": CfgSectionSpec(
                                fields={"half_width": fields["half_width"]}
                            ),
                            "fixed": CfgSectionSpec(
                                fields={"manual_value": fields["manual_value"]}
                            ),
                        },
                    ),
                ),
            )
        },
        {
            "search": CfgSectionValue(
                fields={
                    "mode": DirectValue("auto"),
                    "half_width": DirectValue(1.0),
                    "manual_value": DirectValue(2.0),
                }
            )
        },
    )
    form = CfgFormWidget()
    model = attach_draft(form, schema, ctrl)
    original_root = form._root_widget
    assert isinstance(original_root, TreeCfgWidget)
    monkeypatch.setattr(original_root, "refresh_section", lambda _path: False)
    emitted: list[CfgSchema] = []
    form.schema_changed.connect(emitted.append)

    search = cast(SectionField, model.fields["search"])
    search.fields["mode"].set_value(DirectValue("fixed"))
    form._flush_pending_section_refresh()

    assert form._root_widget is not original_root
    assert emitted == []
    assert form._schema_snapshot_pending is True

    qapp.processEvents()

    assert len(emitted) == 1
    emitted_search = emitted[0].value.fields["search"]
    assert isinstance(emitted_search, CfgSectionValue)
    assert emitted_search.fields["mode"] == DirectValue("fixed")


def test_spec_tooltip_populates_decoration_and_provider_can_override(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )

    class TooltipProvider:
        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            del spec, value
            if path == "gain":
                return FieldDecorationPatch(tooltip="Provider tooltip")
            return None

    schema = section_schema(
        {
            "gain": ScalarSpec(
                label="Gain",
                type=float,
                tooltip="Spec tooltip",
            ),
            "window": SweepSpec(label="Window", tooltip="Sweep tooltip"),
        },
        {
            "gain": DirectValue(1.0),
            "window": SweepValue(start=0.0, stop=1.0, expts=11),
        },
    )
    w = CfgFormWidget(decoration_provider=TooltipProvider())
    attach_draft(w, schema, ctrl)

    assert w.decoration_for_path("gain").tooltip == "Provider tooltip"
    assert w.decoration_for_path("window").tooltip == "Sweep tooltip"
    # Sole tree: tooltips are on QTreeWidgetItem, not ElidedLabel
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    root = w._root_widget
    assert isinstance(root, TreeCfgWidget)
    # Find items for gain and window
    gain_item = None
    window_item = None
    stack = [root._tree.invisibleRootItem()]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        for i in range(cur.childCount()):
            child = cur.child(i)
            if child is None:
                continue
            if child.text(0).startswith("Gain"):
                gain_item = child
            if child.text(0).startswith("Window"):
                window_item = child
            stack.append(child)
    assert gain_item is not None and gain_item.toolTip(0) == "Provider tooltip"
    assert window_item is not None and window_item.toolTip(0) == "Sweep tooltip"


def test_sweep_edge_decoration_disables_only_that_edge(qapp, ctrl):
    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    class StopGeneratedProvider:
        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            del spec, value
            if path == "window.stop":
                return FieldDecorationPatch(
                    enabled=False,
                    tone="muted",
                    badge="generated",
                    tooltip="Stop is generated",
                )
            return None

    schema = section_schema(
        {"window": SweepSpec(label="Window")},
        {"window": SweepValue(start=0.0, stop=10.0, expts=21)},
    )
    w = CfgFormWidget(decoration_provider=StopGeneratedProvider())
    attach_draft(w, schema, ctrl)

    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None
    assert w.decoration_for_path("window.start").enabled is True
    assert w.decoration_for_path("window.stop").enabled is False
    assert sweep_widget._start_widget.isEnabled() is True
    assert sweep_widget._stop_widget.isEnabled() is False
    assert sweep_widget._expts.isEnabled() is True
    labels = {
        label.text(): label.toolTip() for label in sweep_widget.findChildren(QLabel)
    }
    assert labels["stop [generated]"] == "Stop is generated"


def test_choice_section_rejects_unknown_choice_fields():
    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto"]),
    }

    with pytest.raises(RuntimeError, match="unknown field"):
        ChoiceSectionSpec(
            fields=fields,
            bindings=(
                ChoiceBinding(
                    "mode",
                    {
                        "auto": CfgSectionSpec(
                            fields={"missing": ScalarSpec(label="Missing", type=float)}
                        )
                    },
                ),
            ),
        )


def test_choice_section_unknown_selector_value_fast_fails(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "auto_gain": ScalarSpec(label="Auto gain", type=float),
        "manual_gain": ScalarSpec(label="Manual gain", type=float),
    }
    schema = section_schema(
        {
            "search": ChoiceSectionSpec(
                label="Search",
                fields=fields,
                bindings=(
                    ChoiceBinding(
                        "mode",
                        {
                            "auto": CfgSectionSpec(
                                fields={"auto_gain": fields["auto_gain"]}
                            ),
                            "fixed": CfgSectionSpec(
                                fields={"manual_gain": fields["manual_gain"]}
                            ),
                        },
                    ),
                ),
            )
        },
        {
            "search": CfgSectionValue(
                fields={
                    "mode": DirectValue("unknown"),
                    "auto_gain": DirectValue(0.1),
                    "manual_gain": DirectValue(0.2),
                }
            )
        },
    )

    with pytest.raises(ValueError, match="unknown value 'unknown'"):
        attach_draft(CfgFormWidget(), schema, ctrl)


def test_literal_rows_are_hidden_regardless_of_key(qapp, ctrl):
    """All LiteralSpec fields render no widget — discriminators (type/style) and
    adapter lock_literal'd fields (e.g. a sweep-driven freq) alike."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {
            "type": LiteralSpec("pulse", label="Type"),
            # a non-type/style LiteralSpec: the lock_literal scenario
            "freq": LiteralSpec(0.0, label="Freq"),
            "waveform": CfgSectionSpec(
                label="Waveform",
                fields={
                    "style": LiteralSpec("gauss", label="Style"),
                    "sigma": ScalarSpec(label="Sigma", type=float),
                },
            ),
        },
        {
            "waveform": CfgSectionValue(fields={"sigma": DirectValue(1.2)}),
        },
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)

    # Sole tree: hidden literals mean no QTreeWidgetItem for those paths
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    root = w._root_widget
    assert isinstance(root, TreeCfgWidget)
    # Verify tree has Sigma but not Type/Style/Freq
    found = set()
    stack = [root._tree.invisibleRootItem()]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        for i in range(cur.childCount()):
            child = cur.child(i)
            if child is None:
                continue
            found.add(child.text(0))
            stack.append(child)
    assert not any("Type" in txt for txt in found)
    assert not any(txt == "Style" or "Style" in txt for txt in found)
    # Freq is a literal at top level, should be hidden (no item)
    assert not any(txt == "Freq" for txt in found)
    assert any("Sigma" in txt for txt in found)


def test_literal_rows_revealed_by_decoration_use_framed_read_only_value(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )

    class RevealLiteralProvider:
        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            del spec, value
            if path == "freq":
                return FieldDecorationPatch(
                    hidden=False,
                    enabled=False,
                    badge="generated",
                    tooltip="Generated at run time",
                )
            return None

    schema = section_schema(
        {"freq": LiteralSpec(0.0, label="Freq")},
        {},
    )
    w = CfgFormWidget(decoration_provider=RevealLiteralProvider())
    attach_draft(w, schema, ctrl)

    # Sole tree: revealed literal appears as a tree item with generated badge, not ElidedLabel
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    root = w._root_widget
    assert isinstance(root, TreeCfgWidget)
    found = False
    stack = [root._tree.invisibleRootItem()]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        for i in range(cur.childCount()):
            child = cur.child(i)
            if child is None:
                continue
            if "Freq" in child.text(0) and "generated" in child.text(0):
                found = True
                assert child.isDisabled() is True
            stack.append(child)
    assert found
    # Value widget for literal is still a read-only line edit inside tree
    literal_edits = [edit for edit in w.findChildren(QLineEdit) if edit.text() == "0.0"]
    assert len(literal_edits) == 1
    assert literal_edits[0].isReadOnly() is True
    assert literal_edits[0].isEnabled() is False


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


def test_populate_full_fake_freq_schema(qapp, ctrl):
    """Smoke test: FakeFreqAdapter default schema populates and round-trips."""
    from zcu_tools.experiment.v2_gui.measure.adapters.fake.freq import FakeFreqAdapter
    from zcu_tools.gui.cfg import ReferenceSpec
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctx = _make_ctx()
    schema = FakeFreqAdapter().make_default_cfg(ctx)

    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    out = w.read_values()

    for key in ("reps", "rounds", "sweep", "modules"):
        assert key in out.fields, f"missing key: {key}"
    # The simulated resonance moved to the adapter __init__ — no 'model' in cfg.
    assert "model" not in out.fields

    assert isinstance(out.fields["sweep"], CfgSectionValue)
    assert isinstance(out.fields["sweep"].fields["freq"], SweepValue)
    # modules is a CfgSectionValue with readout as ReferenceValue
    modules_val = out.fields["modules"]
    assert isinstance(modules_val, CfgSectionValue)
    # Verify spec has ReferenceSpec for readout
    modules_spec = schema.spec.fields["modules"]
    assert hasattr(modules_spec, "fields")
    readout_spec = modules_spec.fields["readout"]  # type: ignore[union-attr]
    assert isinstance(readout_spec, ReferenceSpec)


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


# ---------------------------------------------------------------------------
# optional ReferenceSpec UI (None option in combo)
# ---------------------------------------------------------------------------


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
