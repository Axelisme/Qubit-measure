"""Shared helper for gated, per-experiment reset module writeback.

Each reset calibration adapter proposes its final reset library module the same
way: gate on whether every canonical calibration path is present, then
build the module from *this* run's ``cfg_snapshot.modules.tested_reset`` template
and overwrite fields from its fixed ParameterContext. Reset calibration runs
are unordered and repeatable, so the proposal is keyed on completeness — not on
being "the last step" — letting any run that has the full calibration emit it.
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


class _HasTestedReset(Protocol):
    # Read-only property (not a bare attr) so the Protocol is covariant: a concrete
    # ``modules`` whose ``tested_reset`` is a specific ResetCfg subtype still
    # matches ``object`` (a mutable attr would be invariant and reject it).
    @property
    def tested_reset(self) -> object: ...


class _HasModules(Protocol):
    """Structural surface of any reset cfg snapshot: a ``modules.tested_reset``.

    Every reset calibration cfg carries the run's tested reset under
    ``modules.tested_reset``; this is the only attribute this helper reads, so the
    Protocol stays as narrow as the access (no whole-cfg type dependency).
    """

    @property
    def modules(self) -> _HasTestedReset: ...


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
            ("pulse_cfg.freq", best_ro_freq),
            ("ro_cfg.ro_freq", best_ro_freq),
            ("pulse_cfg.gain", best_ro_gain),
            (
                "pulse_cfg.waveform.length",
                best_ro_length + READOUT_DPM_PULSE_TAIL_US,
            ),
            ("ro_cfg.ro_length", best_ro_length),
        ),
        role_id="readout_dpm",
    )


def reset_module_writeback_items(
    ctx: SessionEnv,
    cfg_snapshot: _HasModules | None,
    *,
    target: str,
    field_md_map: Sequence[tuple[str, str]],
    desc: str,
) -> list[ModuleWriteback]:
    """A gated ``ModuleWriteback`` proposing the calibrated reset library module.

    ``field_md_map`` pairs fields inside ``tested_reset`` with canonical
    parameter paths such as ``$qubit.module.sideband.pulse_cfg.freq``.
    A missing capture, snapshot, null or absent calibration returns ``[]``.
    Invalid sources, aliases and scalar values propagate without fallback.
    The template remains this run's ``cfg_snapshot.modules.tested_reset``.
    """
    if cfg_snapshot is None or ctx.parameters is None:
        return []
    calibrated_fields: list[tuple[str, float]] = []
    for field_path, parameter_path in field_md_map:
        calibrated = optional_parameter_float(ctx.parameters, parameter_path)
        if calibrated is None:
            return []
        calibrated_fields.append((field_path, calibrated))

    tested_reset = cfg_snapshot.modules.tested_reset
    spec, value = module_cfg_to_value(tested_reset)
    for field_path, calibrated in calibrated_fields:
        value.with_field(field_path, calibrated)

    return [
        ModuleWriteback(
            target_name=target,
            description=desc,
            edit_schema=CfgSchema(spec=spec, value=value),
        )
    ]
