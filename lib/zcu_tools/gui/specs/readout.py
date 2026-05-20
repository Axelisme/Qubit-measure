"""Static CfgSectionSpec definitions for readout module types."""

from __future__ import annotations

from zcu_tools.gui.adapter import CfgSectionSpec, ScalarSpec
from zcu_tools.gui.specs.pulse import PULSE_SPEC

DIRECT_READOUT_SPEC = CfgSectionSpec(
    label="Direct Readout",
    fields={
        "type": ScalarSpec(label="Type", type=str, hidden=True),
        "ro_ch": ScalarSpec(label="RO ch", type=int),
        "ro_freq": ScalarSpec(label="RO Freq (MHz)", type=float, decimals=2),
        "ro_length": ScalarSpec(label="RO length (us)", type=float, decimals=3),
        "trig_offset": ScalarSpec(label="Trig offset (us)", type=float, decimals=3),
    },
)

PULSE_READOUT_SPEC = CfgSectionSpec(
    label="Pulse Readout",
    fields={
        "type": ScalarSpec(label="Type", type=str, hidden=True),
        "pulse_cfg": PULSE_SPEC,
        "ro_cfg": DIRECT_READOUT_SPEC,
    },
)
