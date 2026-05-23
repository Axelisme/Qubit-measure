"""Fresh CfgSectionSpec factories for readout module types."""

from __future__ import annotations

from zcu_tools.gui.adapter import CfgSectionSpec, LiteralSpec, ScalarSpec
from zcu_tools.gui.specs.pulse import make_pulse_spec


def make_direct_readout_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Direct Readout",
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
            "direct_cfg": make_direct_readout_spec(),
            "pulse_cfg": make_pulse_readout_spec(),
        },
    )
