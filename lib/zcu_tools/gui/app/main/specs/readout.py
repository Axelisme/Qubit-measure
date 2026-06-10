"""Fresh CfgSectionSpec factories for readout module types."""

from __future__ import annotations

from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    LiteralSpec,
    ScalarSpec,
    make_default_value,
)

from .pulse import make_pulse_spec


def _inherit_direct_readout(
    old_val: CfgSectionValue, old_spec: CfgSectionSpec
) -> CfgSectionValue | None:
    if old_spec.label != "Pulse Readout":
        return None
    ro_cfg_val = old_val.fields.get("ro_cfg")
    if isinstance(ro_cfg_val, CfgSectionValue):
        return ro_cfg_val
    return None


def _inherit_pulse_readout(
    old_val: CfgSectionValue, old_spec: CfgSectionSpec
) -> CfgSectionValue | None:
    if old_spec.label != "Direct Readout":
        return None
    result = make_default_value(make_pulse_readout_spec())
    result.fields["ro_cfg"] = old_val
    return result


def make_direct_readout_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Direct Readout",
        inherit_hook=_inherit_direct_readout,
        fields={
            "type": LiteralSpec("readout/direct"),
            "ro_ch": ScalarSpec(label="RO ch", type=int),
            "ro_freq": ScalarSpec(label="RO Freq (MHz)", type=float, decimals=2),
            "ro_length": ScalarSpec(label="RO length (us)", type=float, decimals=3),
            "trig_offset": ScalarSpec(label="Trig offset (us)", type=float, decimals=3),
        },
    )


def make_pulse_readout_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Pulse Readout",
        inherit_hook=_inherit_pulse_readout,
        fields={
            "type": LiteralSpec("readout/pulse"),
            "pulse_cfg": make_pulse_spec(),
            "ro_cfg": make_direct_readout_spec(),
        },
    )


def make_readout_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Readout",
        fields={
            "type": LiteralSpec("readout/direct", "readout/pulse"),
            "pulse_cfg": make_pulse_readout_spec(),
            "direct_cfg": make_direct_readout_spec(),
        },
    )
