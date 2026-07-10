"""Utility functions for CfgSchema logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.meta_tool.library import ModuleLibrary

    from ...adapter import CfgSectionSpec, CfgSectionValue, ReferenceSpec


def _spec_value_for_chosen(
    chosen_key: str,
    ref_spec: ReferenceSpec,
    ml: ModuleLibrary | None,
) -> tuple[CfgSectionSpec | None, CfgSectionValue | None]:
    """Find both Spec and initial Value for a chosen key (Custom or Named)."""
    from ...cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

    logger.debug(
        "_spec_value_for_chosen: key=%r allowed_labels=%r",
        chosen_key,
        [s.label for s in ref_spec.allowed],
    )

    # 1. Custom template from 'allowed' list
    if chosen_key.startswith("<Custom:"):
        label = chosen_key[8:-1]
        for spec in ref_spec.allowed:
            if spec.label == label:
                logger.debug("_spec_value_for_chosen: matched custom label=%r", label)
                return spec, None
        raise RuntimeError(f"Unknown custom reference label: {label!r}")

    # 2. Named instance from Library
    if ml:
        if ref_spec.kind == "module":
            store = ml.modules
            converter = module_cfg_to_value
        elif ref_spec.kind == "waveform":
            store = ml.waveforms
            converter = waveform_cfg_to_value
        else:
            raise RuntimeError(f"Unsupported reference kind {ref_spec.kind!r}")
        if chosen_key in store:
            cfg = store[chosen_key]
            logger.debug(
                "_spec_value_for_chosen: matched %s key=%r",
                ref_spec.kind,
                chosen_key,
            )
            _, value = converter(cfg)
            from ...adapter import (
                ReferenceValue,
                align_locked_literals,
                select_ref_value_spec,
            )

            spec = select_ref_value_spec(ref_spec, ReferenceValue(chosen_key, value))
            return spec, align_locked_literals(spec, value)

    raise RuntimeError(f"Unknown library reference: {chosen_key!r}")
