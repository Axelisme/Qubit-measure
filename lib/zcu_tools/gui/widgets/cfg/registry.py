"""Exact factory registry for shared cfg field widgets."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, Self, cast, final, runtime_checkable

from qtpy.QtWidgets import QLineEdit, QWidget  # type: ignore[attr-defined]

from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgField,
    LiteralField,
    ReferenceField,
    ScalarField,
    SectionField,
    SweepField,
)

from .decoration import FieldDecorationProtocol

TextInputEnhancer = Callable[[QLineEdit], object | None]
FieldDecorationResolver = Callable[
    [str, CfgField],
    FieldDecorationProtocol,
]


@runtime_checkable
class FieldWidgetProtocol(Protocol):
    """Runtime surface shared by cfg field widgets."""

    @property
    def field(self) -> CfgField: ...

    def refresh_section(self, path: str) -> bool: ...

    def teardown(self) -> None: ...


@dataclass(frozen=True)
class FieldRenderContext:
    """Immutable presentation context for one field renderer invocation."""

    registry: FrozenFieldRendererRegistry
    path: str = ""
    top_level: bool = False
    field_label_max_width: int | None = None
    decoration_for_path: FieldDecorationResolver | None = None
    text_input_enhancer: TextInputEnhancer | None = None

    def derive(
        self,
        *,
        path: str | None = None,
        top_level: bool | None = None,
    ) -> FieldRenderContext:
        """Return a child context while retaining the same frozen registry."""
        return replace(
            self,
            path=self.path if path is None else path,
            top_level=self.top_level if top_level is None else top_level,
        )


class FieldRenderer(Protocol):
    """Fixed factory contract for every registered field renderer."""

    def __call__(
        self,
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol: ...


_SIGNATURE_FIELD = object()
_SIGNATURE_CONTEXT = object()


@final
class FieldRendererRegistry:
    """Mutable one-shot builder for exact field renderer factories."""

    def __init__(self) -> None:
        self._renderers: dict[type[CfgField], FieldRenderer] = {}
        self._is_frozen = False

    def register(
        self,
        field_type: type[CfgField],
        renderer: FieldRenderer,
    ) -> Self:
        if self._is_frozen:
            raise RuntimeError("FieldRendererRegistry is frozen")
        if not isinstance(field_type, type) or not issubclass(field_type, CfgField):
            raise TypeError("field_type must be a CfgField type")
        if not callable(renderer):
            raise TypeError("renderer must be callable")
        _validate_renderer_signature(renderer)
        if field_type in self._renderers:
            raise ValueError(
                f"Renderer already registered for field type {field_type.__name__}"
            )
        self._renderers[field_type] = renderer
        return self

    def freeze(self) -> FrozenFieldRendererRegistry:
        if self._is_frozen:
            raise RuntimeError("FieldRendererRegistry is already frozen")
        self._is_frozen = True
        return FrozenFieldRendererRegistry(self._renderers)


@final
class FrozenFieldRendererRegistry:
    """Immutable exact-type field renderer lookup and construction boundary."""

    def __init__(
        self,
        renderers: Mapping[type[CfgField], FieldRenderer],
    ) -> None:
        self._renderers = MappingProxyType(dict(renderers))

    def resolve(self, field_or_type: CfgField | type[CfgField]) -> FieldRenderer:
        field_type = (
            field_or_type if isinstance(field_or_type, type) else type(field_or_type)
        )
        try:
            return self._renderers[field_type]
        except KeyError as exc:
            raise TypeError(
                f"No renderer registered for exact field type {field_type.__name__}"
            ) from exc

    def render(
        self,
        field: CfgField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        if context.registry is not self:
            raise ValueError("FieldRenderContext registry does not match renderer")
        renderer = self.resolve(field)
        widget = renderer(field, context)
        if not isinstance(widget, QWidget):
            raise TypeError(
                f"Renderer for {type(field).__name__} returned "
                f"{type(widget).__name__}; expected QWidget-compatible "
                "FieldWidgetProtocol"
            )
        if not isinstance(widget, FieldWidgetProtocol):
            widget.deleteLater()
            raise TypeError(
                f"Renderer for {type(field).__name__} returned "
                f"{type(widget).__name__}; expected FieldWidgetProtocol"
            )
        return widget


def default_cfg_renderers() -> FrozenFieldRendererRegistry:
    """Build a fresh frozen registry containing the standard cfg factories."""
    return (
        FieldRendererRegistry()
        .register(LiteralField, _render_literal)
        .register(ScalarField, _render_scalar)
        .register(SweepField, _render_sweep)
        .register(CenteredSweepField, _render_centered_sweep)
        .register(SectionField, _render_section)
        .register(ReferenceField, _render_reference)
        .freeze()
    )


def _validate_renderer_signature(renderer: FieldRenderer) -> None:
    try:
        signature = inspect.signature(renderer)
        signature.bind(_SIGNATURE_FIELD, _SIGNATURE_CONTEXT)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "renderer must accept exactly the call shape (field, context)"
        ) from exc


def _render_literal(
    field: CfgField,
    context: FieldRenderContext,
) -> FieldWidgetProtocol:
    del context
    from .fields import LiteralWidget

    return LiteralWidget(cast(LiteralField, field))


def _render_scalar(
    field: CfgField,
    context: FieldRenderContext,
) -> FieldWidgetProtocol:
    from .fields import ScalarWidget

    return ScalarWidget(
        cast(ScalarField, field),
        text_input_enhancer=context.text_input_enhancer,
    )


def _render_sweep(
    field: CfgField,
    context: FieldRenderContext,
) -> FieldWidgetProtocol:
    from .fields import SweepWidget

    return SweepWidget(
        cast(SweepField, field),
        path=context.path,
        decoration_for_path=context.decoration_for_path,
        text_input_enhancer=context.text_input_enhancer,
    )


def _render_centered_sweep(
    field: CfgField,
    context: FieldRenderContext,
) -> FieldWidgetProtocol:
    from .fields import CenteredSweepWidget

    return CenteredSweepWidget(
        cast(CenteredSweepField, field),
        text_input_enhancer=context.text_input_enhancer,
    )


def _render_section(
    field: CfgField,
    context: FieldRenderContext,
) -> FieldWidgetProtocol:
    from .fields import SectionWidget

    return SectionWidget(cast(SectionField, field), context=context)


def _render_reference(
    field: CfgField,
    context: FieldRenderContext,
) -> FieldWidgetProtocol:
    from .fields import ReferenceWidget

    return ReferenceWidget(cast(ReferenceField, field), context=context)


__all__ = [
    "FieldDecorationResolver",
    "FieldRenderContext",
    "FieldRenderer",
    "FieldRendererRegistry",
    "FieldWidgetProtocol",
    "FrozenFieldRendererRegistry",
    "TextInputEnhancer",
    "default_cfg_renderers",
]
