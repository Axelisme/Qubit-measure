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
from .structure import (
    TREE_DEPTH_COLORS,
    TreeCfgWidget,
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
    "TREE_DEPTH_COLORS",
    "Tone",
    "TextInputEnhancer",
    "TreeCfgWidget",
    "default_cfg_renderers",
    "default_decoration_for_spec",
]
