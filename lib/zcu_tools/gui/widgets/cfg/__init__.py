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
    FormStructure,
    StructuralAdapter,
    TreeCfgWidget,
    TreeStructure,
    form_structure,
    tree_structure,
)

__all__ = [
    "CfgFormWidget",
    "FieldDecoration",
    "FieldDecorationPatch",
    "FieldDecorationProvider",
    "FieldRenderContext",
    "FieldRenderer",
    "FieldRendererRegistry",
    "FormStructure",
    "FrozenFieldRendererRegistry",
    "StructuralAdapter",
    "TREE_DEPTH_COLORS",
    "Tone",
    "TextInputEnhancer",
    "TreeCfgWidget",
    "TreeStructure",
    "default_cfg_renderers",
    "default_decoration_for_spec",
    "form_structure",
    "tree_structure",
]
