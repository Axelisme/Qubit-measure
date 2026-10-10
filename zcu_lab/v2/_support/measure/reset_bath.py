"""Shared writeback wiring for the bath reset calibration chain."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from zcu_tools.gui.app.measure.adapter import ModuleWriteback, SessionEnv

from zcu_lab.v2._support.measure.writeback_helpers import reset_module_writeback_items

if TYPE_CHECKING:
    from zcu_lab.v2._support.measure.writeback_helpers import _HasModules

# The bath chain proposes two final modules from the same calibrated tested_reset:
# 'reset_bath' (pi/2 phase = ground/max) and 'reset_bath_e' (pi/2 phase =
# excited/min). Each reads cavity freq/gain and phase from its own captured
# bath_g/bath_e slot. All three bath steps (freq_gain / length / phase) propose
# the same pair, so the wiring lives here as the single source.
_BATH_VARIANTS: tuple[tuple[str, str, str], ...] = (
    (
        "reset_bath",
        "bath_g",
        "Reset to Ground with cavity-assisted bath reset",
    ),
    (
        "reset_bath_e",
        "bath_e",
        "Reset to Excited with cavity-assisted bath reset",
    ),
)


def bath_reset_writeback_items(
    ctx: SessionEnv, cfg_snapshot: _HasModules | None
) -> list[ModuleWriteback]:
    """Gated 'reset_bath' + 'reset_bath_e' proposals from a bath calibration run.

    Each variant gates independently on its captured bath_g/bath_e calibration
    paths. Complete values overwrite this run's tested_reset template. Missing
    fields omit that variant; malformed sources and aliases propagate.
    """
    items: list[ModuleWriteback] = []
    for target, slot, desc in _BATH_VARIANTS:
        field_md_map: Sequence[tuple[str, str]] = (
            ("cavity_tone_cfg.freq", f"$qubit.module.{slot}.cavity_tone_cfg.freq"),
            ("cavity_tone_cfg.gain", f"$qubit.module.{slot}.cavity_tone_cfg.gain"),
            ("pi2_cfg.phase", f"$qubit.module.{slot}.pi2_cfg.phase"),
        )
        items.extend(
            reset_module_writeback_items(
                ctx,
                cfg_snapshot,
                target=target,
                field_md_map=field_md_map,
                desc=desc,
            )
        )
    return items
