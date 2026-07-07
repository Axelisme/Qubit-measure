"""Shared helper for gated, per-experiment reset module writeback.

Each reset calibration adapter proposes its final reset library module the same
way: gate on whether every calibration md key the module needs is present, then
build the module from *this* run's ``cfg_snapshot.modules.tested_reset`` template
and overwrite the calibration fields from md. The user's reset calibration runs
are unordered and repeatable, so the proposal is keyed on md completeness — not on
being "the last step" — letting any run that has the full calibration emit it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    ModuleWriteback,
)
from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value
from zcu_tools.program.v2.modules import PulseReadoutCfg

from .ctx_helpers import md_get_float, md_has_key

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext


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


def _resolve_readout_dpm_values(
    ctx: ExpContext, proposed: Mapping[str, float]
) -> tuple[float, float, float] | None:
    values: list[float] = []
    for key in _READOUT_DPM_KEYS:
        if key in proposed:
            values.append(float(proposed[key]))
            continue
        if not md_has_key(ctx, key):
            return None
        values.append(md_get_float(ctx, key, float("nan")))

    best_ro_freq, best_ro_gain, best_ro_length = values
    return best_ro_freq, best_ro_gain, best_ro_length


def _pulse_readout_module_writeback_items(
    cfg_snapshot: _HasReadoutModules | None,
    *,
    target: str,
    desc: str,
    field_updates: Sequence[tuple[str, float]],
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
        )
    ]


def readout_dpm_writeback_items(
    ctx: ExpContext,
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
    return _pulse_readout_module_writeback_items(
        cfg_snapshot,
        target="readout_dpm",
        desc="Optimized readout (DPM)",
        field_updates=(
            ("pulse_cfg.freq", best_ro_freq),
            ("ro_cfg.ro_freq", best_ro_freq),
            ("pulse_cfg.gain", best_ro_gain),
            ("pulse_cfg.waveform.length", best_ro_length + 0.1),
            ("ro_cfg.ro_length", best_ro_length),
        ),
    )


def readout_rf_writeback_items(
    cfg_snapshot: _HasReadoutModules | None,
    *,
    r_f: float,
) -> list[ModuleWriteback]:
    return _pulse_readout_module_writeback_items(
        cfg_snapshot,
        target="readout_rf",
        desc="Readout at fitted resonator frequency",
        field_updates=(
            ("pulse_cfg.freq", float(r_f)),
            ("ro_cfg.ro_freq", float(r_f)),
        ),
    )


def reset_module_writeback_items(
    ctx: ExpContext,
    cfg_snapshot: _HasModules | None,
    *,
    target: str,
    field_md_map: Sequence[tuple[str, str]],
    desc: str,
) -> list[ModuleWriteback]:
    """A gated ``ModuleWriteback`` proposing the calibrated reset library module.

    ``field_md_map`` pairs a dotted field path inside ``tested_reset`` (e.g.
    ``"pulse_cfg.freq"``) with the md key holding its calibrated value (e.g.
    ``"reset_f"``). The proposal is emitted only when *every* md key is present
    (``md_has_key``) and a ``cfg_snapshot`` exists; otherwise it returns ``[]`` so
    the experiment offers only its plain md writeback. When emitted, the module is
    built from ``cfg_snapshot.modules.tested_reset`` (this run's calibrated
    template) and each mapped field is overwritten from md.
    """
    if cfg_snapshot is None:
        return []
    if not all(md_has_key(ctx, md_key) for _, md_key in field_md_map):
        return []

    tested_reset = cfg_snapshot.modules.tested_reset
    spec, value = module_cfg_to_value(tested_reset)
    for field_path, md_key in field_md_map:
        # md_has_key gated every key above, so md_get_float's default is unreachable;
        # pass NaN so a regression (a missing key slipping through) fails loudly
        # downstream rather than silently writing a plausible 0.0.
        value.with_field(field_path, md_get_float(ctx, md_key, float("nan")))

    return [
        ModuleWriteback(
            target_name=target,
            description=desc,
            edit_schema=CfgSchema(spec=spec, value=value),
        )
    ]
