"""Default factories for the ``readout`` role (qubit-state readout).

Two concrete shapes exist: ``make_pulse_readout_default`` (inline pulse + ro_cfg;
references a library ``ro_waveform`` for the pulse waveform when present) and
``make_direct_readout_default`` (bare DirectReadout, no pulse). ``make_readout_default``
is the role's blank — a thin alias to the pulse shape (adapters whose spec only
allows PulseReadout should call ``make_pulse_readout_default`` directly so the chosen
shape is explicit, not implied by the role default). The ref prefers the calibrated
library readouts (readout_dpm / readout_rf).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, overload

from zcu_tools.gui.app.main.adapter import (
    CfgSectionValue,
    ModuleRefValue,
    WaveformRefValue,
)
from zcu_tools.gui.app.main.specs.readout import (
    make_direct_readout_spec,
    make_pulse_readout_spec,
)
from zcu_tools.program.v2.modules import PulseReadoutCfg

from ..ctx_helpers import md_get_float, md_scalar_float, md_scalar_int
from .helpers import (
    make_default_value,
    make_trig_offset,
    patch_pulse_fields,
    patch_ro_cfg_fields,
    select_named_module_value,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext

READOUT_NAMES = ["readout_dpm", "readout_rf", "readout", "res_readout"]


def make_pulse_readout_default(ctx: ExpContext) -> ModuleRefValue:
    """Blank inline pulse-readout (pulse + ro_cfg). References a library
    ``ro_waveform`` for the pulse waveform when present.

    Adapter-specific tuning (gain, ro_length, …) is applied by the caller via
    ``.with_field(...)`` rather than factory parameters.
    """
    r_f = md_scalar_float(ctx, "r_f", 6000.0)
    res_ch = md_scalar_int(ctx, "res_ch", 0)
    ro_ch = md_scalar_int(ctx, "ro_ch", 0)
    trig_offset = make_trig_offset(ctx, trig_expr="timeFly + 0.05", trig_fallback=0.55)

    value = make_default_value(make_pulse_readout_spec())

    pulse_cfg = value.fields.get("pulse_cfg")
    if isinstance(pulse_cfg, CfgSectionValue):
        patch_pulse_fields(pulse_cfg, freq=r_f, ch=res_ch, gain=0.1, length=1.0)
        waveform_ref = pulse_cfg.fields.get("waveform")
        if (
            isinstance(waveform_ref, WaveformRefValue)
            and "ro_waveform" in ctx.ml.waveforms
        ):
            from zcu_tools.gui.app.main.cfg_schemas import waveform_cfg_to_value

            _, wav_val = waveform_cfg_to_value(ctx.ml.waveforms["ro_waveform"])
            pulse_cfg.fields["waveform"] = WaveformRefValue(
                chosen_key="ro_waveform", value=wav_val
            )

    ro_cfg = value.fields.get("ro_cfg")
    if isinstance(ro_cfg, CfgSectionValue):
        patch_ro_cfg_fields(
            ro_cfg, ro_freq=r_f, ro_ch=ro_ch, trig_offset=trig_offset, ro_length=0.9
        )

    return ModuleRefValue("<Custom:Pulse Readout>", value)


def make_direct_readout_default(ctx: ExpContext) -> ModuleRefValue:
    """Blank DirectReadout (bare ro_cfg, no pulse excitation).

    Adapter-specific tuning is applied by the caller via ``.with_field(...)``.
    """
    r_f = md_scalar_float(ctx, "r_f", 6000.0)
    ro_ch = md_scalar_int(ctx, "ro_ch", 0)
    trig_offset = make_trig_offset(ctx, trig_expr="timeFly + 0.05", trig_fallback=0.55)

    value = make_default_value(make_direct_readout_spec())
    patch_ro_cfg_fields(
        value, ro_freq=r_f, ro_ch=ro_ch, trig_offset=trig_offset, ro_length=0.9
    )
    return ModuleRefValue("<Custom:Direct Readout>", value)


def make_readout_default(ctx: ExpContext) -> ModuleRefValue:
    """The readout role's blank — a pulse readout."""
    return make_pulse_readout_default(ctx)


def make_readout_dpm_default(ctx: ExpContext) -> ModuleRefValue:
    """Optimized pulse-readout (``readout_dpm``): a pulse readout seeded from the
    readout-optimization results in the MetaDict — 'best_ro_freq' / 'best_ro_gain'
    / 'best_ro_length' (the outputs of the ro_optimize experiments). The pulse
    window is best_ro_length + 0.1 us, the acquisition window is best_ro_length;
    references a library ``ro_waveform`` for the pulse waveform when present.
    Mirrors the notebook's readout_dpm assembly. Falls back to r_f / 0.1 / 1.0 us
    when the best_ro_* values are absent."""
    r_f = md_get_float(ctx, "r_f", 6000.0)
    best_freq = md_scalar_float(ctx, "best_ro_freq", r_f)
    best_gain = md_get_float(ctx, "best_ro_gain", 0.1)
    best_len = md_get_float(ctx, "best_ro_length", 1.0)
    res_ch = md_scalar_int(ctx, "res_ch", 0)
    ro_ch = md_scalar_int(ctx, "ro_ch", 0)
    trig_offset = make_trig_offset(ctx, trig_expr="timeFly + 0.05", trig_fallback=0.55)

    value = make_default_value(make_pulse_readout_spec())

    pulse_cfg = value.fields.get("pulse_cfg")
    if isinstance(pulse_cfg, CfgSectionValue):
        patch_pulse_fields(
            pulse_cfg, freq=best_freq, ch=res_ch, gain=best_gain, length=best_len + 0.1
        )
        waveform_ref = pulse_cfg.fields.get("waveform")
        if (
            isinstance(waveform_ref, WaveformRefValue)
            and "ro_waveform" in ctx.ml.waveforms
        ):
            from zcu_tools.gui.app.main.cfg_schemas import waveform_cfg_to_value

            _, wav_val = waveform_cfg_to_value(ctx.ml.waveforms["ro_waveform"])
            pulse_cfg.fields["waveform"] = WaveformRefValue(
                chosen_key="ro_waveform", value=wav_val
            )

    ro_cfg = value.fields.get("ro_cfg")
    if isinstance(ro_cfg, CfgSectionValue):
        patch_ro_cfg_fields(
            ro_cfg,
            ro_freq=best_freq,
            ro_ch=ro_ch,
            trig_offset=trig_offset,
            ro_length=best_len,
        )

    return ModuleRefValue("<Custom:Pulse Readout>", value)


@overload
def make_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> ModuleRefValue: ...


@overload
def make_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> ModuleRefValue | None: ...


def make_readout_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = READOUT_NAMES,
    *,
    optional: bool = False,
) -> ModuleRefValue | None:
    """Reference a calibrated library readout (readout_dpm / readout_rf), else
    fall back to the blank inline pulse-readout."""
    selected = select_named_module_value(
        ml=ctx.ml, module_type=PulseReadoutCfg, preferred_names=preferred_names
    )
    if selected is not None:
        return ModuleRefValue(chosen_key=selected.name, value=selected.value)
    if optional:
        return None  # optional ref disabled (ADR-0010)
    return make_readout_default(ctx)
