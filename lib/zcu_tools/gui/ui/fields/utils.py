"""Utility functions for CfgSchema logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict
    from zcu_tools.meta_tool.library import ModuleLibrary
    from .adapter import CfgSectionSpec


def _resolve_channel(text: str, md: Optional[MetaDict]) -> Optional[int]:
    """Resolve a channel string (int or MetaDict key) to a physical channel ID."""
    try:
        return int(text)
    except ValueError:
        pass

    if md is None:
        return None

    try:
        val = getattr(md, text)
        return int(val)
    except (AttributeError, ValueError, TypeError):
        return None


def _spec_value_for_chosen(
    chosen_key: str,
    allowed: list[CfgSectionSpec],
    ml: Optional[ModuleLibrary],
) -> tuple[Optional[CfgSectionSpec], Optional[Any]]:
    """Find both Spec and initial Value for a chosen key (Custom or Named)."""
    from ...cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

    # 1. Custom template from 'allowed' list
    if chosen_key.startswith("<Custom:"):
        label = chosen_key[8:-1]
        for spec in allowed:
            if spec.label == label:
                return spec, None
        return (allowed[0] if allowed else None), None

    # 2. Named instance from Library
    if ml:
        if chosen_key in ml.modules:
            cfg = ml.modules[chosen_key]
            return module_cfg_to_value(cfg)
        if chosen_key in ml.waveforms:
            cfg = ml.waveforms[chosen_key]
            return waveform_cfg_to_value(cfg)

    return None, None
