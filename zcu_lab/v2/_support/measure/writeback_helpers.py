"""Shared pulse-readout proposal mechanics and existing DPM calibration policy.

Reset finishers live in their adapters and materialize the Run-tested cfg
without reading calibration values from the capture.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.measure.adapter import ModuleWriteback
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.program.v2.modules import PulseReadoutCfg

from zcu_lab.v2._support.measure.ctx_helpers import optional_parameter_float

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv


class _HasReadout(Protocol):
    @property
    def readout(self) -> object: ...


class _HasReadoutModules(Protocol):
    @property
    def modules(self) -> _HasReadout: ...


_READOUT_DPM_KEYS = ("best_ro_freq", "best_ro_gain", "best_ro_length")
READOUT_DPM_PULSE_TAIL_US = 0.1


def _resolve_readout_dpm_values(
    ctx: SessionEnv, proposed: Mapping[str, float]
) -> tuple[float, float, float] | None:
    values: list[float] = []
    for key in _READOUT_DPM_KEYS:
        if key in proposed:
            values.append(float(proposed[key]))
            continue
        if ctx.parameters is None:
            return None
        calibrated = optional_parameter_float(ctx.parameters, f"$qubit.ext.{key}")
        if calibrated is None:
            return None
        values.append(calibrated)

    best_ro_freq, best_ro_gain, best_ro_length = values
    if not all(math.isfinite(value) for value in values):
        return None
    return best_ro_freq, best_ro_gain, best_ro_length


def pulse_readout_module_writeback_items(
    cfg_snapshot: _HasReadoutModules | None,
    *,
    target: str,
    desc: str,
    field_updates: Sequence[tuple[str, float]],
    role_id: str | None = None,
) -> list[ModuleWriteback]:
    if cfg_snapshot is None:
        return []

    readout_cfg = cfg_snapshot.modules.readout
    if not isinstance(readout_cfg, PulseReadoutCfg):
        return []

    spec, value = module_cfg_to_value(readout_cfg)
    for field_path, field_value in field_updates:
        value.with_field(field_path, field_value)

    return [
        ModuleWriteback(
            target_name=target,
            description=desc,
            edit_schema=CfgSchema(spec=spec, value=value),
            role_id=role_id,
        )
    ]


def readout_dpm_writeback_items(
    ctx: SessionEnv,
    cfg_snapshot: _HasReadoutModules | None,
    *,
    proposed: Mapping[str, float],
) -> list[ModuleWriteback]:
    if cfg_snapshot is None:
        return []

    resolved = _resolve_readout_dpm_values(ctx, proposed)
    if resolved is None:
        return []

    best_ro_freq, best_ro_gain, best_ro_length = resolved
    return pulse_readout_module_writeback_items(
        cfg_snapshot,
        target="readout_dpm",
        desc="Optimized readout (DPM)",
        field_updates=(
            ("freq", best_ro_freq),
            ("gain", best_ro_gain),
            (
                "waveform.length",
                best_ro_length + READOUT_DPM_PULSE_TAIL_US,
            ),
            ("ro_length", best_ro_length),
        ),
        role_id="readout_dpm",
    )
