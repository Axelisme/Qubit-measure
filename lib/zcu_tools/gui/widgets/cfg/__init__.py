"""Shared Qt widgets for rendering the app-independent cfg model."""

from .decoration import (
    FieldDecoration,
    FieldDecorationPatch,
    FieldDecorationProvider,
    Tone,
    default_decoration_for_spec,
)
from .form import CfgFormWidget
from .registry import (
    FieldRenderContext,
    FieldRenderer,
    FieldRendererRegistry,
    FrozenFieldRendererRegistry,
    TextInputEnhancer,
    default_cfg_renderers,
)

__all__ = [
    "CfgFormWidget",
    "FieldDecoration",
    "FieldDecorationPatch",
    "FieldDecorationProvider",
    "FieldRenderContext",
    "FieldRenderer",
    "FieldRendererRegistry",
    "FrozenFieldRendererRegistry",
    "Tone",
    "TextInputEnhancer",
    "default_cfg_renderers",
    "default_decoration_for_spec",
]
