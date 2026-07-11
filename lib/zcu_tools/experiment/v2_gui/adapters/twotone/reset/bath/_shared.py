"""Shared writeback wiring for the bath reset calibration chain."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from zcu_tools.experiment.v2_gui.adapters._support import reset_module_writeback_items
from zcu_tools.gui.app.main.adapter import ExpContext, ModuleWriteback

if TYPE_CHECKING:
    from zcu_tools.experiment.v2_gui.adapters._support.writeback_helpers import (
        _HasModules,
    )

# The bath chain proposes two final modules from the same calibrated tested_reset:
# 'reset_bath' (pi/2 phase = ground/max) and 'reset_bath_e' (pi/2 phase =
# excited/min). Both also overwrite the calibrated cavity freq/gain from md; only
# the pi2 phase md key differs. All three bath steps (freq_gain / length / phase)
# propose the same pair, so the wiring lives here as the single source.
_CAVITY_FIELD_MD_MAP: tuple[tuple[str, str], ...] = (
    ("cavity_tone_cfg.freq", "bathreset_freq"),
    ("cavity_tone_cfg.gain", "bathreset_gain"),
)

_BATH_VARIANTS: tuple[tuple[str, str, str], ...] = (
    (
        "reset_bath",
        "bathreset_max_phase",
        "Reset to Ground with cavity-assisted bath reset",
    ),
    (
        "reset_bath_e",
        "bathreset_min_phase",
        "Reset to Excited with cavity-assisted bath reset",
    ),
)


def bath_reset_writeback_items(
    ctx: ExpContext, cfg_snapshot: _HasModules | None
) -> list[ModuleWriteback]:
    """Gated 'reset_bath' + 'reset_bath_e' proposals from a bath calibration run.

    Each variant gates on its full md key set (cavity freq/gain + its pi/2 phase)
    independently, so a run that has only one phase calibrated still proposes that
    one. Built from this run's tested_reset template with md overwriting the fields.
    """
    items: list[ModuleWriteback] = []
    for target, phase_key, desc in _BATH_VARIANTS:
        field_md_map: Sequence[tuple[str, str]] = (
            *_CAVITY_FIELD_MD_MAP,
            ("pi2_cfg.phase", phase_key),
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
