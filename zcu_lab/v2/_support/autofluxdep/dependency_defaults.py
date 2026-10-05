"""Default dependency fallbacks shared by autofluxdep nodes."""

from __future__ import annotations

from typing import Any

from zcu_tools.program.v2.modules import PulseCfg


def missing_info_value() -> None:
    return None


def missing_module_value() -> Any | None:
    return None


def is_lowerable_pulse_module(module: Any) -> bool:
    """Whether a resolved drive module is a concrete, lowerable pulse module."""
    if isinstance(module, PulseCfg):
        return True
    if isinstance(module, dict):
        return module.get("type") == "pulse"
    return False
