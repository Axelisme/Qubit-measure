"""Sensible-value ModuleRefValue builders for each module type.

Each function calls make_default_value(spec) to get the structural skeleton,
then patches fields with context-aware reasonable values from ctx.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from zcu_tools.gui.adapter import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ModuleRefValue,
    ScalarValue,
    WaveformRefValue,
    make_default_value,
)

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext
from zcu_tools.gui.specs.pulse import make_pulse_spec
from zcu_tools.gui.specs.readout import (
    make_direct_readout_spec,
    make_pulse_readout_spec,
)
from zcu_tools.gui.specs.reset import (
    make_bath_reset_spec,
    make_none_reset_spec,
    make_pulse_reset_spec,
    make_two_pulse_reset_spec,
)

from .ctx_helpers import (
    md_get_float,
    md_get_int,
    md_has_key,
    md_scalar_float,
    md_scalar_int,
)

# ---------------------------------------------------------------------------
# Internal patch helpers
# ---------------------------------------------------------------------------


def _patch_pulse_fields(
    value: CfgSectionValue,
    *,
    freq: Union[float, ScalarValue],
    ch: Union[int, ScalarValue],
    gain: float,
    length: float,
) -> None:
    """Patch a pulse CfgSectionValue (flat fields) in-place with sensible values."""
    waveform_ref = value.fields.get("waveform")
    if isinstance(waveform_ref, (ModuleRefValue, WaveformRefValue)):
        waveform_ref.value.fields["length"] = DirectValue(length)

    value.fields["ch"] = (
        ch if isinstance(ch, (DirectValue, EvalValue)) else DirectValue(ch)
    )
    value.fields["nqz"] = DirectValue(2)
    value.fields["freq"] = (
        freq if isinstance(freq, (DirectValue, EvalValue)) else DirectValue(freq)
    )
    value.fields["gain"] = DirectValue(gain)


def _patch_ro_cfg_fields(
    value: CfgSectionValue,
    *,
    ro_freq: Union[float, ScalarValue],
    ro_ch: Union[int, ScalarValue],
    trig_offset: Union[float, ScalarValue],
    ro_length: Union[float, ScalarValue] = 0.9,
) -> None:
    """Patch a DirectReadout CfgSectionValue in-place with sensible values."""
    value.fields["ro_freq"] = (
        ro_freq
        if isinstance(ro_freq, (DirectValue, EvalValue))
        else DirectValue(ro_freq)
    )
    value.fields["ro_ch"] = (
        ro_ch if isinstance(ro_ch, (DirectValue, EvalValue)) else DirectValue(ro_ch)
    )
    value.fields["ro_length"] = (
        ro_length
        if isinstance(ro_length, (DirectValue, EvalValue))
        else DirectValue(ro_length)
    )
    value.fields["trig_offset"] = (
        trig_offset
        if isinstance(trig_offset, (DirectValue, EvalValue))
        else DirectValue(trig_offset)
    )


# ---------------------------------------------------------------------------
# Readout defaults
# ---------------------------------------------------------------------------


def make_direct_readout_default(ctx: ExpContext) -> ModuleRefValue:
    r_f = md_scalar_float(ctx, "r_f", 6000.0)
    ro_ch = md_scalar_int(ctx, "ro_ch", 0)
    trig_offset = _make_trig_offset(
        ctx,
        trig_expr="timeFly + 0.05",
        trig_delta=0.05,
        trig_fallback=0.55,
    )

    spec = make_direct_readout_spec()
    value = make_default_value(spec)
    _patch_ro_cfg_fields(value, ro_freq=r_f, ro_ch=ro_ch, trig_offset=trig_offset)
    return ModuleRefValue("<Custom:Direct Readout>", value)


def _make_trig_offset(
    ctx: ExpContext,
    *,
    trig_expr: str,
    trig_delta: float,
    trig_fallback: float,
) -> ScalarValue:
    """Build a trig_offset ScalarValue: EvalValue if timeFly exists, else DirectValue."""
    if md_has_key(ctx, "timeFly"):
        resolved = md_get_float(ctx, "timeFly", trig_fallback - trig_delta) + trig_delta
        return EvalValue(expr=trig_expr, resolved=resolved, error=None)
    return DirectValue(trig_fallback)


def make_pulse_readout_default(
    ctx: ExpContext,
    *,
    gain: float = 0.1,
    ro_length: Union[float, ScalarValue] = 0.9,
    trig_expr: str = "timeFly + 0.05",
    trig_delta: float = 0.05,
    trig_fallback: float = 0.55,
) -> ModuleRefValue:
    r_f = md_scalar_float(ctx, "r_f", 6000.0)
    res_ch = md_scalar_int(ctx, "res_ch", 0)
    ro_ch = md_scalar_int(ctx, "ro_ch", 0)
    trig_offset = _make_trig_offset(
        ctx,
        trig_expr=trig_expr,
        trig_delta=trig_delta,
        trig_fallback=trig_fallback,
    )

    spec = make_pulse_readout_spec()
    value = make_default_value(spec)

    pulse_cfg = value.fields.get("pulse_cfg")
    if isinstance(pulse_cfg, CfgSectionValue):
        _patch_pulse_fields(pulse_cfg, freq=r_f, ch=res_ch, gain=gain, length=1.0)
        waveform_ref = pulse_cfg.fields.get("waveform")
        if (
            isinstance(waveform_ref, WaveformRefValue)
            and "ro_waveform" in ctx.ml.waveforms
        ):
            from zcu_tools.gui.cfg_schemas import waveform_cfg_to_value

            _, wav_val = waveform_cfg_to_value(ctx.ml.waveforms["ro_waveform"])
            waveform_ref = WaveformRefValue(chosen_key="ro_waveform", value=wav_val)
            pulse_cfg.fields["waveform"] = waveform_ref

    ro_cfg = value.fields.get("ro_cfg")
    if isinstance(ro_cfg, CfgSectionValue):
        _patch_ro_cfg_fields(
            ro_cfg,
            ro_freq=r_f,
            ro_ch=ro_ch,
            trig_offset=trig_offset,
            ro_length=ro_length,
        )

    return ModuleRefValue("<Custom:Pulse Readout>", value)


def make_readout_default(ctx: ExpContext) -> ModuleRefValue:
    return make_pulse_readout_default(ctx)


# ---------------------------------------------------------------------------
# Pulse default
# ---------------------------------------------------------------------------


def make_pulse_default(
    ctx: ExpContext, *, gain: float = 0.05, length: float = 0.1
) -> ModuleRefValue:
    q_f = md_get_float(ctx, "q_f", 4000.0)
    qub_ch = md_get_int(ctx, "qub_ch", 0)

    spec = make_pulse_spec()
    value = make_default_value(spec)
    _patch_pulse_fields(value, freq=q_f, ch=qub_ch, gain=gain, length=length)
    return ModuleRefValue("<Custom:Pulse>", value)


# ---------------------------------------------------------------------------
# Reset defaults
# ---------------------------------------------------------------------------


def make_none_reset_default(ctx: ExpContext) -> ModuleRefValue:  # noqa: ARG001
    spec = make_none_reset_spec()
    value = make_default_value(spec)
    return ModuleRefValue("<Custom:None Reset>", value)


def make_pulse_reset_default(ctx: ExpContext) -> ModuleRefValue:
    q_f = md_get_float(ctx, "q_f", 4000.0)
    qub_ch = md_get_int(ctx, "qub_ch", 0)

    spec = make_pulse_reset_spec()
    value = make_default_value(spec)

    pulse_cfg = value.fields.get("pulse_cfg")
    if isinstance(pulse_cfg, CfgSectionValue):
        _patch_pulse_fields(pulse_cfg, freq=q_f, ch=qub_ch, gain=0.2, length=1.0)

    return ModuleRefValue("<Custom:Pulse Reset>", value)


def make_two_pulse_reset_default(ctx: ExpContext) -> ModuleRefValue:
    q_f = md_get_float(ctx, "q_f", 4000.0)
    qub_ch = md_get_int(ctx, "qub_ch", 0)

    spec = make_two_pulse_reset_spec()
    value = make_default_value(spec)

    for key in ("pulse1_cfg", "pulse2_cfg"):
        pulse_cfg = value.fields.get(key)
        if isinstance(pulse_cfg, CfgSectionValue):
            _patch_pulse_fields(pulse_cfg, freq=q_f, ch=qub_ch, gain=0.2, length=1.0)

    return ModuleRefValue("<Custom:Two-Pulse Reset>", value)


def make_bath_reset_default(ctx: ExpContext) -> ModuleRefValue:
    r_f = md_scalar_float(ctx, "r_f", 6000.0)
    q_f = md_get_float(ctx, "q_f", 4000.0)
    res_ch = md_get_int(ctx, "res_ch", 0)
    qub_ch = md_get_int(ctx, "qub_ch", 0)

    spec = make_bath_reset_spec()
    value = make_default_value(spec)

    cav = value.fields.get("cavity_tone_cfg")
    if isinstance(cav, CfgSectionValue):
        _patch_pulse_fields(cav, freq=r_f, ch=res_ch, gain=0.1, length=1.0)

    qub = value.fields.get("qubit_tone_cfg")
    if isinstance(qub, CfgSectionValue):
        _patch_pulse_fields(qub, freq=q_f, ch=qub_ch, gain=0.2, length=1.0)

    pi2 = value.fields.get("pi2_cfg")
    if isinstance(pi2, CfgSectionValue):
        _patch_pulse_fields(pi2, freq=q_f, ch=qub_ch, gain=0.2, length=1.0)

    return ModuleRefValue("<Custom:Bath Reset>", value)


def make_reset_default(ctx: ExpContext) -> ModuleRefValue:
    return make_pulse_reset_default(ctx)
