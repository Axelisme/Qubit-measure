"""Utility functions for CfgSchema logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.meta_tool.library import ModuleLibrary

    from .adapter import CfgSectionSpec


def _spec_value_for_chosen(
    chosen_key: str,
    allowed: list[CfgSectionSpec],
    ml: Optional[ModuleLibrary],
) -> tuple[Optional[CfgSectionSpec], Optional[Any]]:
    """Find both Spec and initial Value for a chosen key (Custom or Named)."""
    from ...cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

    logger.debug(
        "_spec_value_for_chosen: key=%r allowed_labels=%r",
        chosen_key,
        [s.label for s in allowed],
    )

    # 1. Custom template from 'allowed' list
    if chosen_key.startswith("<Custom:"):
        label = chosen_key[8:-1]
        for spec in allowed:
            if spec.label == label:
                logger.debug("_spec_value_for_chosen: matched custom label=%r", label)
                return spec, None
        raise RuntimeError(f"Unknown custom reference label: {label!r}")

    # 2. Named instance from Library
    if ml:
        if chosen_key in ml.modules:
            cfg = ml.modules[chosen_key]
            logger.debug("_spec_value_for_chosen: matched module key=%r", chosen_key)
            return module_cfg_to_value(cfg)
        if chosen_key in ml.waveforms:
            cfg = ml.waveforms[chosen_key]
            logger.debug("_spec_value_for_chosen: matched waveform key=%r", chosen_key)
            return waveform_cfg_to_value(cfg)

    raise RuntimeError(f"Unknown library reference: {chosen_key!r}")
