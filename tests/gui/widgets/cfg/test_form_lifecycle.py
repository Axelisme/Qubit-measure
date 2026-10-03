"""Form attachment, subscriptions and publication lifecycle."""

from __future__ import annotations

from typing import Literal, cast

import pytest
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
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
    FieldRenderContext,
    FieldRenderer,
    FieldRendererRegistry,
    FrozenFieldRendererRegistry,
    default_cfg_renderers,
)
from zcu_tools.gui.widgets.cfg.registry import FieldWidgetProtocol

from tests.gui.widgets.cfg._form_support import attach_draft, section_schema

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
