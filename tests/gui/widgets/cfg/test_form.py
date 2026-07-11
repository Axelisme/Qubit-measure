"""Tests for shared CfgFormWidget populate and read-values behavior."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
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
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.widgets.cfg import (
    FieldRenderContext,
    FieldRenderer,
    FieldRendererRegistry,
    FrozenFieldRendererRegistry,
    default_cfg_renderers,
)
from zcu_tools.gui.widgets.cfg.registry import FieldWidgetProtocol

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ctrl():
    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = MagicMock()
    c.get_current_ml.return_value = MagicMock()
    c.list_arb_waveforms.return_value = []
    return c


def _schema(spec_fields: dict, value_fields: dict) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields=spec_fields),
        value=CfgSectionValue(fields=value_fields),
    )


def _attach(w, schema: CfgSchema, ctrl):
    """Build a caller-owned draft, attach the widget, and return its root field."""
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    return draft.root


def _scalar_field(
    ctrl: MagicMock, spec: ScalarSpec, initial_val: object
) -> ScalarField:
    bindings = MeasureCfgBindings(ctrl)
    return ScalarField(
        spec,
        bindings.evaluate_expression,
        bindings.provide_options,
        initial_val,
    )


_RENDERED_FIELD_TYPES = (
    LiteralField,
    ScalarField,
    SweepField,
    CenteredSweepField,
    SectionField,
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
    return builder.freeze()


def _registry_with_section_factory(
    renderer: FieldRenderer,
) -> FrozenFieldRendererRegistry:
    return _registry_with_factories({SectionField: renderer})


def _marked_section_factory(
    marker: str,
    calls: list[tuple[CfgField, FieldRenderContext]],
) -> FieldRenderer:
    def factory(
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        from zcu_tools.gui.widgets.cfg.fields import SectionWidget

        calls.append((field, context))
        widget = SectionWidget(cast(SectionField, field), context=context)
        widget.setObjectName(marker)
        return widget

    return factory


def _make_ctx():
    from zcu_tools.gui.app.main.adapter import ExpContext

    return ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None)


# ---------------------------------------------------------------------------
# schema_to_dict — SweepValue
# ---------------------------------------------------------------------------


def test_sweep_value_uses_expts_as_canonical():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = _schema(
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
    schema = _schema(
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
    from qtpy.QtWidgets import QComboBox
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctrl.list_arb_waveforms.return_value = ["asset_a", "asset_b"]
    schema = _schema(
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
    w = CfgFormWidget()
    model = _attach(w, schema, ctrl)

    combo = w.findChild(QComboBox)
    assert combo is not None
    assert [combo.itemText(i) for i in range(combo.count())] == ["asset_a", "asset_b"]
    assert combo.currentIndex() == -1
    assert not w.is_valid()

    combo.setCurrentIndex(1)

    field = cast(ScalarField, model.fields["data"])
    value = field.get_value()
    assert isinstance(value, DirectValue)
    assert value.value == "asset_b"
    assert w.is_valid()


def test_arb_waveform_data_choice_allows_empty_initial_value(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctrl.list_arb_waveforms.return_value = ["asset_a"]
    schema = _schema(
        {
            "data": ScalarSpec(
                label="Data key",
                type=str,
                choices_source="arb_waveforms",
            )
        },
        {"data": DirectValue("")},
    )
    w = CfgFormWidget()
    model = _attach(w, schema, ctrl)

    combo = w.findChild(QComboBox)
    assert combo is not None
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "asset_a"]
    assert combo.currentIndex() == 0
    assert w.is_valid()

    field = cast(ScalarField, model.fields["data"])
    value = field.get_value()
    assert isinstance(value, DirectValue)
    assert value.value == ""


def test_dynamic_choice_renders_inactive_current_value_but_remains_invalid(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctrl.list_arb_waveforms.return_value = []
    schema = _schema(
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
    _attach(form, schema, ctrl)

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


def test_grouped_field_renders_in_collapsed_subsection(qapp, ctrl):
    from zcu_tools.gui.cfg import make_default_value
    from zcu_tools.gui.widgets.cfg import FieldRenderContext, default_cfg_renderers
    from zcu_tools.gui.widgets.cfg.fields import (
        SectionWidget,
        _CollapsibleSection,
    )

    spec = CfgSectionSpec(
        fields={
            "reps": ScalarSpec(label="Reps", type=int),
            "mixer_freq": ScalarSpec(
                label="Mixer freq", type=float, optional=True, group="Advanced"
            ),
        }
    )
    field = (
        MeasureCfgBindings(ctrl)
        .new_draft(CfgSchema(spec, make_default_value(spec)))
        .root
    )
    registry = default_cfg_renderers()
    w = cast(
        SectionWidget,
        registry.render(
            field,
            FieldRenderContext(registry=registry, top_level=True),
        ),
    )

    # Both fields get widgets (grouping is presentation-only, not a value change).
    assert set(w._child_widgets) == {"reps", "mixer_freq"}

    # A collapsed "Advanced" sub-section was created for the grouped field.
    advanced = [
        s
        for s in w.findChildren(_CollapsibleSection)
        if s._header_label is not None and "Advanced" in s._header_label.text()
    ]
    assert len(advanced) == 1
    assert advanced[0]._toggle_btn is not None
    assert not advanced[0]._toggle_btn.isChecked()  # collapsed by default


def test_form_propagates_renderer_registry_through_reference_subtree(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget, default_cfg_renderers
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget, SectionWidget

    inner_spec = CfgSectionSpec(
        label="Inner",
        fields={"value": ScalarSpec(label="Value", type=int)},
    )
    inner_value = CfgSectionValue(fields={"value": DirectValue(1)})
    schema = _schema(
        {"ref": ReferenceSpec(kind="module", allowed=[inner_spec])},
        {
            "ref": ReferenceValue(
                chosen_key="<Custom:Inner>",
                value=inner_value,
            )
        },
    )
    renderers = default_cfg_renderers()
    form = CfgFormWidget(renderers=renderers)

    _attach(form, schema, ctrl)

    root = form._root_widget
    assert isinstance(root, SectionWidget)
    reference = root._child_widgets["ref"]
    assert isinstance(reference, ReferenceWidget)
    assert root._context.registry is renderers
    assert reference._context.registry is renderers
    assert isinstance(reference._sub_widget, SectionWidget)
    assert reference._sub_widget._context.registry is renderers


def test_custom_section_factory_renders_root_nested_and_reference_subtree(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget, SectionWidget

    nested_spec = CfgSectionSpec(
        label="Nested",
        fields={"nested_value": ScalarSpec(label="Nested value", type=int)},
    )
    reference_spec = CfgSectionSpec(
        label="Reference inner",
        fields={"ref_value": ScalarSpec(label="Reference value", type=int)},
    )
    schema = _schema(
        {
            "nested": nested_spec,
            "reference": ReferenceSpec(kind="module", allowed=[reference_spec]),
        },
        {
            "nested": CfgSectionValue(
                fields={"nested_value": DirectValue(1)},
            ),
            "reference": ReferenceValue(
                chosen_key="<Custom:Reference inner>",
                value=CfgSectionValue(fields={"ref_value": DirectValue(2)}),
            ),
        },
    )
    calls: list[tuple[CfgField, FieldRenderContext]] = []
    registry = _registry_with_section_factory(
        _marked_section_factory("custom-section", calls)
    )
    form = CfgFormWidget(renderers=registry)

    _attach(form, schema, ctrl)

    root = cast(SectionWidget, form._root_widget)
    nested = cast(SectionWidget, root._child_widgets["nested"])
    reference = cast(ReferenceWidget, root._child_widgets["reference"])
    reference_subtree = cast(SectionWidget, reference._sub_widget)
    assert root.objectName() == "custom-section"
    assert nested.objectName() == "custom-section"
    assert reference_subtree.objectName() == "custom-section"
    assert {(context.path, context.top_level) for _, context in calls} == {
        ("", True),
        ("nested", False),
        ("reference", True),
    }
    assert all(context.registry is registry for _, context in calls)


def test_custom_reference_factory_renders_actual_widget(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget, SectionWidget

    calls: list[tuple[CfgField, FieldRenderContext]] = []

    def reference_factory(
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        calls.append((field, context))
        widget = ReferenceWidget(cast(ReferenceField, field), context=context)
        widget.setObjectName("custom-reference")
        return widget

    inner_spec = CfgSectionSpec(
        label="Inner",
        fields={"value": ScalarSpec(label="Value", type=int)},
    )
    schema = _schema(
        {"reference": ReferenceSpec(kind="module", allowed=[inner_spec])},
        {
            "reference": ReferenceValue(
                chosen_key="<Custom:Inner>",
                value=CfgSectionValue(fields={"value": DirectValue(1)}),
            )
        },
    )
    registry = _registry_with_factories({ReferenceField: reference_factory})
    form = CfgFormWidget(renderers=registry)

    _attach(form, schema, ctrl)

    root = cast(SectionWidget, form._root_widget)
    reference = cast(ReferenceWidget, root._child_widgets["reference"])
    assert reference.objectName() == "custom-reference"
    assert [(context.path, context.registry) for _, context in calls] == [
        ("reference", registry)
    ]


def test_two_forms_render_with_isolated_custom_factories(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"value": ScalarSpec(label="Value", type=int)},
        {"value": DirectValue(1)},
    )
    first_calls: list[tuple[CfgField, FieldRenderContext]] = []
    second_calls: list[tuple[CfgField, FieldRenderContext]] = []
    first = CfgFormWidget(
        renderers=_registry_with_section_factory(
            _marked_section_factory("first-form", first_calls)
        )
    )
    second = CfgFormWidget(
        renderers=_registry_with_section_factory(
            _marked_section_factory("second-form", second_calls)
        )
    )

    _attach(first, schema, ctrl)
    _attach(second, schema, ctrl)

    assert first._root_widget is not None
    assert second._root_widget is not None
    assert first._root_widget.objectName() == "first-form"
    assert second._root_widget.objectName() == "second-form"
    assert len(first_calls) == 1
    assert len(second_calls) == 1


def test_scalar_widget_minimum_width_reduced(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget

    spec = ScalarSpec(label="Name", type=str)
    w = make_scalar_widget(spec, "demo")
    assert w.minimumWidth() == 20


def test_scalar_widget_eval_mode_shows_resolved_ghost(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    field = _scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("r_f"),
    )

    w = ScalarWidget(field)
    assert w._ghost is not None
    assert w._ghost.text() == "= 6000.0"


def test_scalar_widget_eval_mode_marks_unresolved_red(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    ctrl.get_current_md.return_value = MetaDict()
    field = _scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("missing"),
    )

    w = ScalarWidget(field)
    assert w._ghost is not None
    assert w._ghost.text() == "= ?"
    assert "red" in w._ghost.styleSheet()


def test_measure_cfg_form_value_source_resolves_on_space_in_eval_input(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]
    from zcu_tools.gui.app.main.ui.cfg_binding import make_value_source_input_enhancer
    from zcu_tools.gui.session.value_lookup import ValueInfo
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    ctrl.read_value_source.return_value = (
        ValueInfo("device.flux.value", float, "device:flux"),
        0.125,
    )
    schema = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue("r_f")},
    )
    form = CfgFormWidget(text_input_enhancer=make_value_source_input_enhancer(ctrl))
    root = _attach(form, schema, ctrl)
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
    from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    field = _scalar_field(
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
    from qtpy.QtWidgets import QDoubleSpinBox  # type: ignore[attr-defined]
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    ctrl.get_current_md.return_value = MetaDict()
    field = _scalar_field(
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
    assert w._mode == "direct"
    spin = w.findChild(QDoubleSpinBox)
    assert spin is not None
    assert spin.value() == pytest.approx(0.0)


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
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {
            "reps": ScalarSpec(label="Reps", type=int),
            "freq": ScalarSpec(label="Freq", type=float),
        },
        {
            "reps": DirectValue(100),
            "freq": DirectValue(6.0),
        },
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    assert out.fields["reps"].value == 100  # type: ignore[union-attr]
    assert out.fields["freq"].value == pytest.approx(6.0)  # type: ignore[union-attr]


def test_attach_bad_renderer_return_leaves_draft_callbacks_empty(qapp, ctrl):
    from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    def bad_factory(
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        del field, context
        return cast(FieldWidgetProtocol, QWidget())

    schema = _schema(
        {"value": ScalarSpec(label="Value", type=int)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget(
        renderers=_registry_with_section_factory(bad_factory),
    )

    with pytest.raises(TypeError, match="expected FieldWidgetProtocol"):
        form.attach(draft)

    assert draft.on_change._callbacks == []
    assert draft.on_validity_changed._callbacks == []
    assert form._draft is None
    assert form._root_widget is None
    assert form._field_decorations == {}


def test_attach_factory_exception_leaves_draft_callbacks_empty(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    def failing_factory(
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        del field, context
        raise RuntimeError("factory exploded")

    schema = _schema(
        {"value": ScalarSpec(label="Value", type=int)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget(
        renderers=_registry_with_section_factory(failing_factory),
    )

    with pytest.raises(RuntimeError, match="factory exploded"):
        form.attach(draft)

    assert draft.on_change._callbacks == []
    assert draft.on_validity_changed._callbacks == []
    assert form._draft is None
    assert form._root_widget is None


def test_detach_and_reattach_validity_subscription_emits_once(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"value": ScalarSpec(label="Value", type=int, required=True)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    validity: list[bool] = []
    form.validity_changed.connect(validity.append)

    form.attach(draft)
    assert draft.on_change._callbacks == [form._on_draft_changed]
    assert draft.on_validity_changed._callbacks == [form._on_draft_validity_changed]
    assert validity == [True]

    form.detach()
    assert draft.on_change._callbacks == []
    assert draft.on_validity_changed._callbacks == []

    form.attach(draft)
    assert draft.on_change._callbacks == [form._on_draft_changed]
    assert draft.on_validity_changed._callbacks == [form._on_draft_validity_changed]
    assert validity == [True, True]

    value_field = cast(ScalarField, draft.root.fields["value"])
    value_field.set_value(None)
    assert validity == [True, True, False]


def test_set_editing_enabled_keeps_scroll_area_enabled(qapp, ctrl):
    from qtpy.QtWidgets import QScrollArea, QSpinBox  # type: ignore[attr-defined]
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    w = CfgFormWidget()
    model = _attach(w, schema, ctrl)
    try:
        scroll = w.findChild(QScrollArea)
        assert scroll is not None
        assert w.isEnabled()
        assert scroll.isEnabled()

        assert w._root_widget is not None
        spin = w._root_widget.findChild(QSpinBox)
        assert spin is not None
        assert spin.isEnabled()

        w.set_editing_enabled(False)

        assert w.isEnabled()
        assert scroll.isEnabled()
        assert w._root_widget is not None and not w._root_widget.isEnabled()
        assert not spin.isEnabled()

        draft = w._draft
        assert draft is not None
        w.detach()
        w.attach(draft)

        assert w.isEnabled()
        assert scroll.isEnabled()
        assert w._root_widget is not None and not w._root_widget.isEnabled()
        reattached_spin = w._root_widget.findChild(QSpinBox)
        assert reattached_spin is not None
        assert not reattached_spin.isEnabled()

        w.set_editing_enabled(True)

        assert w._root_widget.isEnabled()
        assert reattached_spin.isEnabled()
    finally:
        w.detach()
        model.teardown()


def test_cfg_form_reflects_model_external_refresh(qapp, ctrl):
    """The widget reflects expression refreshes from its attached draft."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    schema = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue("r_f")},
    )
    w = CfgFormWidget()
    emitted = []
    w.schema_changed.connect(emitted.append)
    model = _attach(w, schema, ctrl)

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

    schema = _schema(
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
        nested = cast(SectionField, draft.root.fields["nested"])
        reps = cast(ScalarField, nested.fields["reps"])

        reps.set_value(11)
        reps.set_value(12)
        reps.set_value(13)

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

    schema = _schema(
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
        value = cast(ScalarField, draft.root.fields["value"])
        value.set_value(None)
        value.set_value(2)
        value.set_value(None)

        assert validity == [True, False, True, False]
        assert schemas == []
        assert form._schema_snapshot_timer.isActive()

        qapp.processEvents()

        assert len(schemas) == 1
        assert not form._schema_snapshot_timer.isActive()
    finally:
        form.detach()
        draft.close()


def test_detach_drops_pending_schema_and_reattach_can_schedule(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"value": ScalarSpec(label="Value", type=int)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    schemas: list[CfgSchema] = []
    form.schema_changed.connect(schemas.append)
    form.attach(draft)
    value = cast(ScalarField, draft.root.fields["value"])

    try:
        value.set_value(2)
        form.detach()
        qapp.processEvents()

        assert schemas == []
        assert not form._schema_snapshot_timer.isActive()

        form.attach(draft)
        value.set_value(3)
        qapp.processEvents()

        assert len(schemas) == 1
        assert schemas[0].value.fields["value"] == DirectValue(3)
    finally:
        form.detach()
        draft.close()


def test_close_drops_pending_schema_and_reattach_can_schedule(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"value": ScalarSpec(label="Value", type=int)},
        {"value": DirectValue(1)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    schemas: list[CfgSchema] = []
    form.schema_changed.connect(schemas.append)
    form.attach(draft)
    form.show()
    value = cast(ScalarField, draft.root.fields["value"])

    try:
        value.set_value(2)
        form.close()
        qapp.processEvents()

        assert schemas == []
        assert form._draft is None
        assert not form._schema_snapshot_timer.isActive()

        form.attach(draft)
        value.set_value(3)
        qapp.processEvents()

        assert len(schemas) == 1
        assert schemas[0].value.fields["value"] == DirectValue(3)
    finally:
        form.detach()
        draft.close()


def test_cfg_form_does_not_subscribe_bus(qapp, ctrl):
    """Attach/detach never registers an EventBus subscription."""
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": DirectValue(6000.0)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    _attach(w, schema, ctrl)  # re-attach swaps models cleanly

    bus = ctrl.get_bus.return_value
    assert bus._subs == {} or all(not subs for subs in bus._subs.values())


def test_read_schema_returns_cfg_schema(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(10)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_schema()
    assert isinstance(out, CfgSchema)
    assert out.spec is schema.spec


def test_read_values_does_not_mutate_original(qapp, ctrl):
    from qtpy.QtWidgets import QSpinBox  # type: ignore[attr-defined]
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    spin = w.findChild(QSpinBox)
    assert spin is not None
    spin.setValue(999)

    out = w.read_values()
    assert out.fields["reps"].value == 999  # type: ignore[union-attr]
    assert schema.value.fields["reps"].value == 100  # type: ignore[union-attr]


def test_populate_sweep_field_round_trip(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=5.8, stop=6.2, expts=201)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.start == pytest.approx(5.8)
    assert sv.stop == pytest.approx(6.2)
    assert sv.expts == 201
    assert sv.step == pytest.approx(0.002)


def test_populate_centered_sweep_field_round_trip(qapp, ctrl):
    from qtpy.QtWidgets import QLabel, QSizePolicy
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import CenteredSweepWidget

    schema = _schema(
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
    w = CfgFormWidget()
    model = _attach(w, schema, ctrl)
    sweep_widget = w.findChild(CenteredSweepWidget)
    assert sweep_widget is not None

    field = cast(CenteredSweepField, model.fields["f"])
    assert field.center_field.spec.editable is False
    assert sweep_widget._center_widget.isEnabled() is False
    for value_widget in (
        sweep_widget._center_widget,
        sweep_widget._span,
        sweep_widget._expts,
        sweep_widget._step,
    ):
        assert (
            value_widget.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
        )
    labels = {label.text(): label for label in sweep_widget.findChildren(QLabel)}
    center_label = labels["center [generated]"]
    span_label = labels["span"]
    assert center_label.toolTip() == "Generated center"
    center_cell = center_label.parentWidget()
    span_cell = span_label.parentWidget()
    assert center_cell is not None
    assert span_cell is not None
    pair_row = center_cell.parentWidget()
    assert pair_row is span_cell.parentWidget()
    assert pair_row is not None
    pair_row.resize(801, pair_row.sizeHint().height())
    qapp.processEvents()
    assert abs(center_cell.width() - span_cell.width()) <= 1

    sweep_widget._span.setValue(120.0)
    sweep_widget._expts.setValue(121)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, CenteredSweepValue)
    assert sv.center == pytest.approx(0.0)
    assert sv.span == pytest.approx(120.0)
    assert sv.expts == 121
    assert sv.step == pytest.approx(1.0)

    sweep_widget._span.setValue(0.0)
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, CenteredSweepValue)
    assert sv.span == pytest.approx(120.0)
    assert sweep_widget._span.value() == pytest.approx(120.0)


def test_populate_sweep_field_step_preserved(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.1)


def test_sweep_widget_step_change_recomputes_expts_and_stop(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    sweep_widget._step.setValue(0.2)
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.expts == 6
    assert sv.stop == pytest.approx(1.0)
    assert sv.step == pytest.approx(0.2)


def test_sweep_widget_non_step_change_recomputes_step(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    sweep_widget._expts.setValue(5)
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.25)


def test_sweep_widget_start_supports_eval_mode(qapp, ctrl):
    from zcu_tools.gui.cfg import EvalValue
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    cast(SweepField, sweep_widget._field).start_field.set_value(
        EvalValue(expr="r_f - 1", resolved=5999.0)
    )
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert isinstance(sv.start, EvalValue)
    assert sv.start.expr == "r_f - 1"


def test_populate_nested_section_round_trip(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {
            "inner": CfgSectionSpec(
                fields={"gain": ScalarSpec(label="Gain", type=float)}
            )
        },
        {"inner": CfgSectionValue(fields={"gain": DirectValue(0.05)})},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    inner = out.fields["inner"]
    assert isinstance(inner, CfgSectionValue)
    assert inner.fields["gain"].value == pytest.approx(0.05)  # type: ignore[union-attr]


def test_nested_sections_render_without_outer_duplicate_label(qapp, ctrl):
    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
        {
            "inner": CfgSectionSpec(
                label="Inner",
                fields={"gain": ScalarSpec(label="Gain", type=float)},
            )
        },
        {"inner": CfgSectionValue(fields={"gain": DirectValue(0.05)})},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    labels = [label.text() for label in w.findChildren(QLabel)]
    assert "Inner:" not in labels
    assert "<b>Inner</b>" in labels
    assert "Gain:" in labels


def test_choice_section_renders_only_active_choice_fields(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half_width": ScalarSpec(label="Half width", type=float),
        "decay": ScalarSpec(label="Decay", type=float),
        "manual_value": ScalarSpec(label="Manual", type=float),
    }
    schema = _schema(
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
    model = _attach(w, schema, ctrl)

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
    from zcu_tools.gui.widgets.cfg.fields import SectionWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half_width": ScalarSpec(label="Half width", type=float),
        "manual_value": ScalarSpec(label="Manual", type=float),
    }
    schema = _schema(
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
    w = CfgFormWidget()
    model = _attach(w, schema, ctrl)
    root_widget = w._root_widget
    assert isinstance(root_widget, SectionWidget)
    search_widget = root_widget._child_widgets["search"]
    stable_widget = root_widget._child_widgets["stable"]
    assert isinstance(search_widget, SectionWidget)
    assert "half_width" in search_widget._child_widgets

    search = model.fields["search"]
    assert isinstance(search, SectionField)
    search.fields["mode"].set_value(DirectValue("fixed"))
    w.decoration_paths()

    assert w._root_widget is root_widget
    assert root_widget._child_widgets["search"] is search_widget
    assert root_widget._child_widgets["stable"] is stable_widget
    assert "half_width" not in search_widget._child_widgets
    assert "manual_value" in search_widget._child_widgets


def test_choice_refresh_fallback_preserves_pending_schema_snapshot(
    qapp, ctrl, monkeypatch: pytest.MonkeyPatch
):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SectionWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half_width": ScalarSpec(label="Half width", type=float),
        "manual_value": ScalarSpec(label="Manual", type=float),
    }
    schema = _schema(
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
    model = _attach(form, schema, ctrl)
    original_root = form._root_widget
    assert isinstance(original_root, SectionWidget)
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


def test_decoration_provider_refresh_rebuilds_only_affected_section(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )
    from zcu_tools.gui.widgets.cfg.fields import SectionWidget

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

    schema = _schema(
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
    _attach(w, schema, ctrl)
    root_widget = w._root_widget
    assert isinstance(root_widget, SectionWidget)
    group_widget = root_widget._child_widgets["group"]
    stable_widget = root_widget._child_widgets["stable"]

    w.set_decoration_provider(BadgeProvider("generated"))

    assert w._root_widget is root_widget
    assert root_widget._child_widgets["group"] is group_widget
    assert root_widget._child_widgets["stable"] is stable_widget
    assert w.decoration_for_path("group.value").badge == "generated"


def test_spec_tooltip_populates_decoration_and_provider_can_override(qapp, ctrl):
    from qtpy.QtWidgets import QWidget
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )
    from zcu_tools.gui.widgets.cfg.fields import ElidedLabel

    class TooltipProvider:
        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            del spec, value
            if path == "gain":
                return FieldDecorationPatch(tooltip="Provider tooltip")
            return None

    schema = _schema(
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
    _attach(w, schema, ctrl)

    assert w.decoration_for_path("gain").tooltip == "Provider tooltip"
    assert w.decoration_for_path("window").tooltip == "Sweep tooltip"
    labels = {
        getattr(label, "_full_text"): label.toolTip()
        for label in w.findChildren(ElidedLabel)
    }
    assert labels["Gain:"] == "Provider tooltip"
    assert labels["Window:"] == "Sweep tooltip"
    value_tooltips = {
        child.toolTip()
        for child in w.findChildren(QWidget)
        if child.toolTip() and not isinstance(child, ElidedLabel)
    }
    assert "Provider tooltip" in value_tooltips
    assert "Sweep tooltip" in value_tooltips


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

    schema = _schema(
        {"window": SweepSpec(label="Window")},
        {"window": SweepValue(start=0.0, stop=10.0, expts=21)},
    )
    w = CfgFormWidget(decoration_provider=StopGeneratedProvider())
    _attach(w, schema, ctrl)

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
    schema = _schema(
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
        _attach(CfgFormWidget(), schema, ctrl)


def test_literal_rows_are_hidden_regardless_of_key(qapp, ctrl):
    """All LiteralSpec fields render no widget — discriminators (type/style) and
    adapter lock_literal'd fields (e.g. a sweep-driven freq) alike."""
    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = _schema(
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
    _attach(w, schema, ctrl)

    labels = [label.text() for label in w.findChildren(QLabel)]
    assert "Type:" not in labels
    assert "Style:" not in labels
    assert "Freq:" not in labels  # locked field hidden too
    assert "Sigma:" in labels


def test_literal_rows_revealed_by_decoration_use_framed_read_only_value(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )
    from zcu_tools.gui.widgets.cfg.fields import ElidedLabel

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

    schema = _schema(
        {"freq": LiteralSpec(0.0, label="Freq")},
        {},
    )
    w = CfgFormWidget(decoration_provider=RevealLiteralProvider())
    _attach(w, schema, ctrl)

    labels = [
        label
        for label in w.findChildren(ElidedLabel)
        if getattr(label, "_full_text") == "Freq [generated]:"
    ]
    assert labels
    assert labels[0].isEnabled() is False

    literal_edits = [edit for edit in w.findChildren(QLineEdit) if edit.text() == "0.0"]
    assert len(literal_edits) == 1
    assert literal_edits[0].isReadOnly() is True
    assert literal_edits[0].isEnabled() is False


def test_module_ref_toggle_sits_left_of_combo_and_controls_subsection(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox, QHBoxLayout, QToolButton
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

    custom_spec = CfgSectionSpec(
        label="Pulse Shape",
        fields={
            "type": LiteralSpec("pulse"),
            "gain": ScalarSpec(label="Gain", type=float),
        },
    )
    schema = _schema(
        {"pulse": ReferenceSpec(kind="module", label="Pulse", allowed=[custom_spec])},
        {
            "pulse": ReferenceValue(
                chosen_key="<Custom:Pulse Shape>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.25)}),
            )
        },
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    w.show()

    ref_widget = w.findChild(ReferenceWidget)
    assert ref_widget is not None

    root_layout = ref_widget.layout()
    assert root_layout is not None
    header_item = root_layout.itemAt(0)
    assert header_item is not None
    header = cast(QHBoxLayout, header_item.layout())
    expand_item = header.itemAt(0)
    combo_item = header.itemAt(1)
    assert expand_item is not None
    assert combo_item is not None
    assert isinstance(expand_item.widget(), QToolButton)
    assert isinstance(combo_item.widget(), QComboBox)
    assert ref_widget._sub_container.isVisible() is True

    ref_widget._expand_btn.click()
    assert ref_widget._sub_container.isVisible() is False

    ref_widget._expand_btn.click()
    assert ref_widget._sub_container.isVisible() is True


def test_waveform_ref_toggle_sits_left_of_combo(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox, QHBoxLayout, QToolButton
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

    custom_spec = CfgSectionSpec(
        label="Gaussian",
        fields={
            "style": LiteralSpec("gauss"),
            "sigma": ScalarSpec(label="Sigma", type=float),
        },
    )
    schema = _schema(
        {
            "waveform": ReferenceSpec(
                kind="waveform", label="Waveform", allowed=[custom_spec]
            )
        },
        {
            "waveform": ReferenceValue(
                chosen_key="<Custom:Gaussian>",
                value=CfgSectionValue(fields={"sigma": DirectValue(0.5)}),
            )
        },
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    ref_widget = w.findChild(ReferenceWidget)
    assert ref_widget is not None
    root_layout = ref_widget.layout()
    assert root_layout is not None
    header_item = root_layout.itemAt(0)
    assert header_item is not None
    header = cast(QHBoxLayout, header_item.layout())
    expand_item = header.itemAt(0)
    combo_item = header.itemAt(1)
    assert expand_item is not None
    assert combo_item is not None
    assert isinstance(expand_item.widget(), QToolButton)
    assert isinstance(combo_item.widget(), QComboBox)


def test_cfg_form_does_not_wrap_module_ref_row(qapp, ctrl):
    from qtpy.QtWidgets import QFormLayout
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SectionWidget

    custom_spec = CfgSectionSpec(
        label="Long Custom Module Name",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    schema = _schema(
        {"pulse": ReferenceSpec(kind="module", label="Pulse", allowed=[custom_spec])},
        {
            "pulse": ReferenceValue(
                chosen_key="<Custom:Long Custom Module Name>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.25)}),
            )
        },
    )
    w = CfgFormWidget()
    w.resize(520, 480)
    _attach(w, schema, ctrl)

    section = w.findChild(SectionWidget)
    assert section is not None
    assert (
        section._container.form.rowWrapPolicy()
        == QFormLayout.RowWrapPolicy.DontWrapRows
    )


def test_populate_module_ref_field_round_trip(qapp, ctrl):
    from zcu_tools.gui.cfg import ReferenceSpec, ReferenceValue
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    allowed_spec = CfgSectionSpec(
        label="Pulse",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "mod": ReferenceSpec(
                    kind="module", allowed=[allowed_spec], label="Module"
                )
            }
        ),
        value=CfgSectionValue(
            fields={
                "mod": ReferenceValue(
                    chosen_key="<Custom:Pulse>",
                    value=CfgSectionValue(fields={"gain": DirectValue(0.5)}),
                )
            }
        ),
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    mod = out.fields["mod"]
    assert isinstance(mod, ReferenceValue)
    assert mod.chosen_key == "<Custom:Pulse>"
    assert mod.value.fields["gain"].value == pytest.approx(0.5)  # type: ignore[union-attr]


def test_populate_full_fake_freq_schema(qapp, ctrl):
    """Smoke test: FakeFreqAdapter default schema populates and round-trips."""
    from zcu_tools.experiment.v2_gui.adapters.fake.freq import FakeFreqAdapter
    from zcu_tools.gui.cfg import ReferenceSpec
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctx = _make_ctx()
    schema = FakeFreqAdapter().make_default_cfg(ctx)

    w = CfgFormWidget()
    _attach(w, schema, ctrl)
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


def test_section_widget_no_header(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import FieldRenderContext, default_cfg_renderers
    from zcu_tools.gui.widgets.cfg.fields import SectionWidget

    spec = CfgSectionSpec(
        label="TestSection",
        fields={"val": ScalarSpec(label="Val", type=int)},
    )
    val = CfgSectionValue(fields={"val": DirectValue(10)})
    field = MeasureCfgBindings(ctrl).new_draft(CfgSchema(spec, val)).root

    # A regular nested section owns a collapsible header.
    renderers = default_cfg_renderers()
    w1 = cast(
        SectionWidget,
        renderers.render(
            field,
            FieldRenderContext(
                registry=renderers,
                path="section",
                top_level=False,
            ),
        ),
    )
    assert w1._container._toggle_btn is not None
    assert w1._container._header_label is not None

    # A reference subtree boundary omits its duplicate section header.
    w2 = cast(
        SectionWidget,
        renderers.render(
            field,
            FieldRenderContext(
                registry=renderers,
                path="reference",
                top_level=True,
            ),
        ),
    )
    assert w2._container._toggle_btn is None
    assert w2._container._header_label is None


def test_module_ref_widget_modified_label_and_no_overwrite(qapp, ctrl):
    from typing import Any, cast

    from qtpy.QtWidgets import QDoubleSpinBox
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget
    from zcu_tools.meta_tool import ModuleLibrary

    ml = ModuleLibrary()
    ml.modules["my_pulse"] = cast(Any, {"type": "readout/direct", "ro_freq": 7000.0})
    ctrl.get_current_ml.return_value = ml

    from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value

    lib_spec, lib_val = module_cfg_to_value(
        {"type": "readout/direct", "ro_freq": 7000.0}
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "mod": ReferenceSpec(kind="module", allowed=[lib_spec], label="Module")
            }
        ),
        value=CfgSectionValue(
            fields={
                "mod": ReferenceValue(
                    chosen_key="my_pulse",
                    value=lib_val,
                )
            }
        ),
    )

    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    w.show()

    ref_widget = w.findChild(ReferenceWidget)
    assert ref_widget is not None
    assert ref_widget._expand_btn.isChecked() is False
    assert ref_widget._sub_container.isVisible() is False
    # 1. Initially unmodified
    assert ref_widget._combo.currentText() == "Lib: my_pulse"
    assert cast(ReferenceField, ref_widget._field).is_modified() is False

    # 2. Simulate user edits the inner value via spinbox
    spin = ref_widget.findChild(QDoubleSpinBox)
    assert spin is not None
    spin.setValue(8000.0)

    # Verify is_modified is True and combobox text has (modified) suffix
    assert cast(ReferenceField, ref_widget._field).is_modified() is True
    assert ref_widget._combo.currentText() == "Lib: my_pulse (modified)"

    # 3. Trigger MD_CHANGED and verify it does not overwrite modified value
    from zcu_tools.gui.session.events import MdChangedPayload
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    ctrl.get_bus.return_value.emit(MdChangedPayload(md=md))

    # Should stay as user modified (8000.0), not library default (7000.0)
    mod_val = w.read_values().fields["mod"]
    assert isinstance(mod_val, ReferenceValue)
    freq_val = mod_val.value.fields["ro_freq"]
    assert isinstance(freq_val, DirectValue)
    assert freq_val.value == 8000.0

    # Verify both modified and clean items are present in combo box
    items_list = [
        ref_widget._combo.itemText(i) for i in range(ref_widget._combo.count())
    ]
    assert "Lib: my_pulse (modified)" in items_list
    assert "Revert to Lib: my_pulse" in items_list

    # 4. Select the clean item to revert modifications
    clean_idx = -1
    for i in range(ref_widget._combo.count()):
        if ref_widget._combo.itemText(i) == "Revert to Lib: my_pulse":
            clean_idx = i
            break
    assert clean_idx >= 0
    ref_widget._combo.setCurrentIndex(clean_idx)

    assert cast(ReferenceField, ref_widget._field).is_modified() is False
    mod_val2 = w.read_values().fields["mod"]
    assert isinstance(mod_val2, ReferenceValue)
    freq_val2 = mod_val2.value.fields["ro_freq"]
    assert isinstance(freq_val2, DirectValue)
    assert freq_val2.value == 7000.0


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


def test_optional_module_ref_renders_none_option(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

    schema = _make_optional_module_ref_schema(enabled=True)
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    module_widgets = w.findChildren(ReferenceWidget)
    assert len(module_widgets) >= 1
    mw = module_widgets[0]

    # None option should be at index 0
    assert mw._combo.itemData(0) == ReferenceWidget._NONE_KEY


def test_optional_module_ref_select_none_disables_sub(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget

    schema = _make_optional_module_ref_schema(enabled=True)
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    module_widgets = w.findChildren(ReferenceWidget)
    mw = module_widgets[0]
    field = cast(ReferenceField, mw._field)

    assert field.is_enabled is True

    # Select the None option
    none_idx = mw._combo.findData(ReferenceWidget._NONE_KEY)
    assert none_idx == 0
    mw._combo.setCurrentIndex(none_idx)

    assert field.is_enabled is False
    assert not mw._sub_container.isEnabled()


def test_module_ref_missing_library_shows_red_badge_and_invalid(qapp, ctrl):
    """A LINKED ref to an absent library key shows the red missing-ref badge and
    is invalid (recoverable — re-adding the name re-links it)."""
    from zcu_tools.gui.cfg import LiteralSpec
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import ReferenceWidget
    from zcu_tools.meta_tool import ModuleLibrary

    pulse_spec = CfgSectionSpec(
        label="Pulse",
        fields={
            "type": LiteralSpec("pulse"),
            "gain": ScalarSpec(label="Gain", type=float),
        },
    )
    schema = _schema(
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

    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    w.show()

    ref_widget = w.findChild(ReferenceWidget)
    assert ref_widget is not None
    field = cast(ReferenceField, ref_widget._field)
    assert field.has_missing_library_ref() is True
    assert field.is_valid() is False
    assert ref_widget._missing_ref_hint.isVisible() is True
    assert "missing_pulse" in ref_widget._missing_ref_hint.text()
