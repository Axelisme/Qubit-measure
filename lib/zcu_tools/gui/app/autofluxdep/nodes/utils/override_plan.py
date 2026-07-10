"""Readable override-plan declarations for autofluxdep node builders."""

from __future__ import annotations

from typing import Self

from zcu_tools.gui.app.autofluxdep.cfg import (
    OverrideMode,
    OverridePath,
    OverridePlan,
    module_override_paths,
)

PULSE_MODULE_LEAF_PATHS: tuple[str, ...] = (
    "type",
    "ch",
    "nqz",
    "freq",
    "gain",
    "waveform",
)

READOUT_FALLBACK_LEAF_PATHS: tuple[str, ...] = (
    "pulse_cfg.freq",
    "pulse_cfg.gain",
    "pulse_cfg.waveform.length",
    "ro_cfg.ro_freq",
    "ro_cfg.ro_length",
)


class NodeOverridePlan:
    """Linear builder for a node's run-time Default cfg override plan."""

    def __init__(self) -> None:
        self._paths: list[OverridePath] = []

    def pulse_module_dependency(
        self,
        module_name: str,
        *,
        source: str | None = None,
        reason: str | None = None,
    ) -> Self:
        label = module_name.replace("_", " ")
        self._paths.extend(
            module_override_paths(
                prefix=f"modules.{module_name}",
                leaf_paths=PULSE_MODULE_LEAF_PATHS,
                source=source or f"{module_name} module dependency",
                reason=reason
                or f"{label} is resolved from workflow/module-library dependency",
            )
        )
        return self

    def readout_dependency(
        self,
        *,
        source: str = "readout module dependency",
        reason: str = "readout module is resolved from workflow/module-library dependency",
    ) -> Self:
        self._paths.extend(
            module_override_paths(
                prefix="modules.readout",
                leaf_paths=READOUT_FALLBACK_LEAF_PATHS,
                source=source,
                reason=reason,
                mode="fallback",
            )
        )
        return self

    def generated_if(
        self,
        enabled: bool,
        path: str,
        *,
        source: str,
        reason: str,
        mode: OverrideMode = "all_points",
    ) -> Self:
        if enabled:
            self._paths.append(OverridePath(path, mode, source, reason))
        return self

    def build(self) -> OverridePlan:
        return OverridePlan(tuple(self._paths))


__all__ = [
    "NodeOverridePlan",
    "PULSE_MODULE_LEAF_PATHS",
    "READOUT_FALLBACK_LEAF_PATHS",
]
