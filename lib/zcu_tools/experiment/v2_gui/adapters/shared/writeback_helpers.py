"""Shared helper for gated, per-experiment reset module writeback.

Each reset calibration adapter proposes its final reset library module the same
way: gate on whether every calibration md key the module needs is present, then
build the module from *this* run's ``cfg_snapshot.modules.tested_reset`` template
and overwrite the calibration fields from md. The user's reset calibration runs
are unordered and repeatable, so the proposal is keyed on md completeness — not on
being "the last step" — letting any run that has the full calibration emit it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    ModuleWriteback,
)
from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value

from .ctx_helpers import md_get_float, md_has_key

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext


def reset_module_writeback_items(
    ctx: ExpContext,
    cfg_snapshot: object | None,
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

    tested_reset = cfg_snapshot.modules.tested_reset  # type: ignore[attr-defined]
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
