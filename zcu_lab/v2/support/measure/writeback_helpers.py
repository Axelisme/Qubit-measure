"""Shared pulse-readout proposal mechanics and existing DPM calibration policy.

Reset finishers live in their adapters and materialize the Run-tested cfg
without reading calibration values from the capture.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from zcu_tools.cfg_model import module_spec_from_raw
from zcu_tools.gui.app.measure.adapter import ModuleWriteback
from zcu_tools.program.v2.modules import AbsModuleCfg, PulseReadoutCfg

from zcu_lab.v2.support.measure.ctx_helpers import optional_parameter_float

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv


class _HasReadout(Protocol):
    @property
    def readout(self) -> object: ...


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


def module_writeback(
    module: AbsModuleCfg,
    *,
    target: str,
    description: str,
    field_updates: Sequence[tuple[str, float]] = (),
    role_id: str | None = None,
) -> ModuleWriteback:
    """Propose the persistent Spec of a Run-time module with fitted fields replaced.

    module is a materialized module from the Run cfg snapshot and is not
    modified; its child pulses and waveforms become inline Specs. Each
    (dotted path, value) in field_updates replaces that field, for example
    ("pulse_cfg.waveform.length", 1.2), before validation, so a swept source
    value can be replaced by its fitted scalar. A path through a missing
    section raises KeyError; a result that is not a valid persistent module
    Spec raises ValueError. target, description and role_id fill the item.
    """
    raw: dict[str, Any] = module.to_dict()
    for path, value in field_updates:
        *sections, leaf = path.split(".")
        node = raw
        for section in sections:
            node = node[section]
        node[leaf] = value
    return ModuleWriteback(
        target_name=target,
        description=description,
        proposal=module_spec_from_raw(raw),
        role_id=role_id,
    )


def pulse_readout_module_writeback_items(
    cfg_snapshot: _HasReadout | None,
    *,
    target: str,
    desc: str,
    field_updates: Sequence[tuple[str, float]],
    role_id: str | None = None,
) -> list[ModuleWriteback]:
    if cfg_snapshot is None:
        return []

    readout_cfg = cfg_snapshot.readout
    if not isinstance(readout_cfg, PulseReadoutCfg):
        return []

    return [
        module_writeback(
            readout_cfg,
            target=target,
            description=desc,
            field_updates=field_updates,
            role_id=role_id,
        )
    ]


def readout_dpm_writeback_items(
    ctx: SessionEnv,
    cfg_snapshot: _HasReadout | None,
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
