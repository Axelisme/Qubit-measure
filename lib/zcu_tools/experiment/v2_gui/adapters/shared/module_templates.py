from __future__ import annotations

import logging

from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionValue,
    ChannelValue,
    DirectValue,
    EvalValue,
    WaveformRefValue,
    make_default_value,
)
from zcu_tools.gui.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.specs.readout import make_pulse_readout_spec
from zcu_tools.gui.specs.waveform import (
    make_const_waveform_spec,
    make_flat_top_waveform_spec,
)

logger = logging.getLogger(__name__)


def update_readout_value_frequency(
    readout_value: CfgSectionValue,
    freq: float,
) -> CfgSectionValue:
    fields = dict(readout_value.fields)

    ro_freq = fields.get("ro_freq")
    if isinstance(ro_freq, (DirectValue, EvalValue)):
        fields["ro_freq"] = DirectValue(freq)

    pulse_cfg = fields.get("pulse_cfg")
    if isinstance(pulse_cfg, CfgSectionValue):
        pulse_fields = dict(pulse_cfg.fields)
        pulse_freq = pulse_fields.get("freq")
        if isinstance(pulse_freq, (DirectValue, EvalValue)):
            pulse_fields["freq"] = DirectValue(freq)
        fields["pulse_cfg"] = CfgSectionValue(fields=pulse_fields)

    ro_cfg = fields.get("ro_cfg")
    if isinstance(ro_cfg, CfgSectionValue):
        ro_fields = dict(ro_cfg.fields)
        ro_cfg_freq = ro_fields.get("ro_freq")
        if isinstance(ro_cfg_freq, (DirectValue, EvalValue)):
            ro_fields["ro_freq"] = DirectValue(freq)
        fields["ro_cfg"] = CfgSectionValue(fields=ro_fields)

    return CfgSectionValue(fields=fields)


def make_readout_edit_template(
    readout: object | None,
    *,
    freq: float,
    pulse_ch: int,
    ro_ch: int,
) -> CfgSchema:
    if readout is not None:
        try:
            spec, readout_value = module_cfg_to_value(readout)
            value = update_readout_value_frequency(readout_value, freq)
            return CfgSchema(spec=spec, value=value)
        except Exception as exc:
            logger.warning("make_readout_edit_template failed: %s", exc)

    return make_pulse_readout_edit_template(
        pulse_ch=pulse_ch,
        pulse_freq=freq,
        ro_ch=ro_ch,
    )


def make_pulse_readout_edit_template(
    *,
    pulse_ch: int,
    pulse_freq: float,
    ro_ch: int,
) -> CfgSchema:
    const_value = make_default_value(make_const_waveform_spec())
    const_value.fields["length"] = DirectValue(1.0)

    pulse_value = CfgSectionValue(
        fields={
            "waveform": WaveformRefValue(
                chosen_key="<Custom:Const>",
                value=const_value,
            ),
            "ch": ChannelValue(chosen=pulse_ch, resolved=None),
            "nqz": DirectValue(2),
            "freq": DirectValue(pulse_freq),
            "phase": DirectValue(0.0),
            "gain": DirectValue(0.2),
            "pre_delay": DirectValue(0.0),
            "post_delay": DirectValue(0.0),
        }
    )
    ro_value = CfgSectionValue(
        fields={
            "ro_ch": ChannelValue(chosen=ro_ch, resolved=None),
            "ro_freq": DirectValue(pulse_freq),
            "ro_length": DirectValue(0.9),
            "trig_offset": DirectValue(0.335),
        }
    )
    value = CfgSectionValue(fields={"pulse_cfg": pulse_value, "ro_cfg": ro_value})
    return CfgSchema(spec=make_pulse_readout_spec(), value=value)


def make_flat_top_waveform_edit_template(*, length: float) -> CfgSchema:
    raise_value = CfgSectionValue(
        fields={
            "style": DirectValue("cosine"),
            "length": DirectValue(0.1),
        }
    )
    value = CfgSectionValue(
        fields={
            "style": DirectValue("flat_top"),
            "length": DirectValue(length),
            "raise_waveform": WaveformRefValue(
                chosen_key="<Custom:Cosine>",
                value=raise_value,
            ),
        }
    )
    return CfgSchema(spec=make_flat_top_waveform_spec(), value=value)
