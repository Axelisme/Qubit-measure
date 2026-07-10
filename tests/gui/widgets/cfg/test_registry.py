"""Tests for the instance-owned exact cfg renderer registry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest
from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]
from zcu_tools.gui.cfg import ScalarSpec
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


def _first_renderer(
    field: CfgField,
    context: FieldRenderContext,
) -> FieldWidgetProtocol:
    raise AssertionError(f"renderer should not be called for {field!r} in {context!r}")


def _second_renderer(
    field: CfgField,
    context: FieldRenderContext,
) -> FieldWidgetProtocol:
    raise AssertionError(f"renderer should not be called for {field!r} in {context!r}")


def test_register_is_chainable_and_resolve_accepts_type_or_field() -> None:
    builder = FieldRendererRegistry()

    assert builder.register(ScalarField, _first_renderer) is builder
    frozen = builder.freeze()
    field = ScalarField(
        ScalarSpec(label="Value", type=int),
        lambda expression: 0,
        None,
        1,
    )

    assert frozen.resolve(ScalarField) is _first_renderer
    assert frozen.resolve(field) is _first_renderer


@pytest.mark.parametrize(
    "renderer",
    [
        lambda: None,
        lambda field: field,
        lambda field, context, required: (field, context, required),
    ],
)
def test_bad_signature_is_rejected_at_registration(renderer: object) -> None:
    with pytest.raises(TypeError, match=r"call shape \(field, context\)"):
        FieldRendererRegistry().register(
            ScalarField,
            cast(FieldRenderer, renderer),
        )


def test_noncallable_renderer_is_rejected() -> None:
    with pytest.raises(TypeError, match="renderer must be callable"):
        FieldRendererRegistry().register(ScalarField, cast(Any, object()))


def test_duplicate_registration_fast_fails() -> None:
    builder = FieldRendererRegistry().register(ScalarField, _first_renderer)

    with pytest.raises(ValueError, match="already registered.*ScalarField"):
        builder.register(ScalarField, _second_renderer)


def test_missing_renderer_fast_fails() -> None:
    frozen = FieldRendererRegistry().freeze()

    with pytest.raises(TypeError, match="exact field type ScalarField"):
        frozen.resolve(ScalarField)


def test_resolve_has_no_inheritance_fallback() -> None:
    class DerivedScalarField(ScalarField):
        pass

    frozen = FieldRendererRegistry().register(ScalarField, _first_renderer).freeze()

    with pytest.raises(TypeError, match="exact field type DerivedScalarField"):
        frozen.resolve(DerivedScalarField)


def test_builder_rejects_mutation_and_second_freeze_after_freeze() -> None:
    builder = FieldRendererRegistry().register(ScalarField, _first_renderer)
    builder.freeze()

    with pytest.raises(RuntimeError, match="is frozen"):
        builder.register(LiteralField, _second_renderer)
    with pytest.raises(RuntimeError, match="already frozen"):
        builder.freeze()


def test_freeze_captures_a_mapping_snapshot() -> None:
    source: dict[type[CfgField], FieldRenderer] = {ScalarField: _first_renderer}
    frozen = FrozenFieldRendererRegistry(source)

    source[ScalarField] = _second_renderer

    assert frozen.resolve(ScalarField) is _first_renderer


def test_registry_instances_are_isolated() -> None:
    first = FieldRendererRegistry().register(ScalarField, _first_renderer).freeze()
    second = FieldRendererRegistry().register(ScalarField, _second_renderer).freeze()

    assert first.resolve(ScalarField) is _first_renderer
    assert second.resolve(ScalarField) is _second_renderer


def test_render_context_is_immutable_presentation_state() -> None:
    registry = FieldRendererRegistry().register(ScalarField, _first_renderer).freeze()
    context = FieldRenderContext(registry=registry, path="root", top_level=True)

    assert {field.name for field in fields(context)} == {
        "registry",
        "path",
        "top_level",
        "field_label_max_width",
        "decoration_for_path",
        "text_input_enhancer",
    }
    with pytest.raises(FrozenInstanceError):
        context.path = "mutated"  # type: ignore[misc]
    child = context.derive(path="root.child", top_level=False)
    assert child.registry is registry
    assert child.path == "root.child"
    assert child.top_level is False


def test_default_factory_returns_fresh_frozen_complete_registries() -> None:
    first = default_cfg_renderers()
    second = default_cfg_renderers()

    assert isinstance(first, FrozenFieldRendererRegistry)
    assert isinstance(second, FrozenFieldRendererRegistry)
    assert first is not second
    for field_type in (
        LiteralField,
        ScalarField,
        SweepField,
        CenteredSweepField,
        SectionField,
        ReferenceField,
    ):
        assert callable(first.resolve(field_type))
        assert callable(second.resolve(field_type))


def test_render_rejects_non_qwidget_result(qapp) -> None:  # noqa: ARG001
    class ProtocolOnly:
        def __init__(self, field: CfgField) -> None:
            self._field = field

        @property
        def field(self) -> CfgField:
            return self._field

        def refresh_section(self, path: str) -> bool:
            del path
            return False

        def teardown(self) -> None:
            pass

    def factory(
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        del context
        return ProtocolOnly(field)

    frozen = FieldRendererRegistry().register(ScalarField, factory).freeze()
    field = ScalarField(
        ScalarSpec(label="Value", type=int),
        lambda expression: 0,
        None,
        1,
    )

    with pytest.raises(TypeError, match="QWidget-compatible FieldWidgetProtocol"):
        frozen.render(field, FieldRenderContext(registry=frozen))


def test_render_rejects_qwidget_without_field_widget_protocol(qapp) -> None:  # noqa: ARG001
    def factory(
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        del field, context
        return cast(FieldWidgetProtocol, QWidget())

    frozen = FieldRendererRegistry().register(ScalarField, factory).freeze()
    field = ScalarField(
        ScalarSpec(label="Value", type=int),
        lambda expression: 0,
        None,
        1,
    )

    with pytest.raises(TypeError, match="expected FieldWidgetProtocol"):
        frozen.render(field, FieldRenderContext(registry=frozen))


def test_render_rejects_context_from_another_registry() -> None:
    first = FieldRendererRegistry().register(ScalarField, _first_renderer).freeze()
    second = FieldRendererRegistry().register(ScalarField, _second_renderer).freeze()
    field = ScalarField(
        ScalarSpec(label="Value", type=int),
        lambda expression: 0,
        None,
        1,
    )

    with pytest.raises(ValueError, match="registry does not match"):
        first.render(field, FieldRenderContext(registry=second))


def test_forms_build_fresh_default_registry_instances(qapp) -> None:  # noqa: ARG001
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    first = CfgFormWidget()
    second = CfgFormWidget()

    assert first._renderers is not second._renderers
