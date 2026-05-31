"""Shared writeback-item builders (avoid per-adapter copy-paste)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from zcu_tools.gui.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
    WritebackItem,
)

from .ctx_helpers import md_get_float
from .spec_helpers import schema_from_module

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext


def make_onetone_freq_writeback_items(
    readout: object, freq: float, fwhm: float, ctx: ExpContext
) -> Sequence[WritebackItem]:
    """Writeback items for a one-tone resonator-frequency fit (real + fake share).

    ``readout`` is the run snapshot's ``modules.readout`` (statically a
    PulseReadoutCfg). Proposes r_f / rf_w into MetaDict and an updated readout
    module + probe waveform length into the ModuleLibrary. No defensive fallback
    — a structural mismatch fast-fails here rather than silently producing an
    unrelated cfg.
    """
    wav_len = md_get_float(ctx, "res_probe_len", 5.0)
    proposed_readout = readout.with_updates(  # type: ignore[attr-defined]
        pulse_cfg=readout.pulse_cfg.with_updates(freq=freq),  # type: ignore[attr-defined]
        ro_cfg=readout.ro_cfg.with_updates(ro_freq=freq),  # type: ignore[attr-defined]
    )
    proposed_waveform = readout.pulse_cfg.waveform.with_updates(  # type: ignore[attr-defined]
        length=wav_len
    )
    return [
        MetaDictWriteback(
            target_name="r_f",
            description="Resonator frequency (MHz)",
            proposed_value=freq,
        ),
        MetaDictWriteback(
            target_name="rf_w",
            description="Resonator linewidth FWHM (MHz)",
            proposed_value=fwhm,
        ),
        ModuleWriteback(
            target_name="readout_rf",
            description="readout_rf module config",
            edit_schema=schema_from_module(proposed_readout),
        ),
        WaveformWriteback(
            target_name="ro_waveform",
            description="ro_waveform length config",
            edit_schema=schema_from_module(proposed_waveform),
        ),
    ]
