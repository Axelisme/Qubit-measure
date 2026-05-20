"""Static CfgSectionSpec definitions for reset module types."""

from __future__ import annotations

from zcu_tools.gui.adapter import CfgSectionSpec, ScalarSpec
from zcu_tools.gui.specs.pulse import PULSE_SPEC

NONE_RESET_SPEC = CfgSectionSpec(
    label="None Reset",
    fields={
        "type": ScalarSpec(label="Type", type=str, hidden=True),
    },
)

PULSE_RESET_SPEC = CfgSectionSpec(
    label="Pulse Reset",
    fields={
        "type": ScalarSpec(label="Type", type=str, hidden=True),
        "pulse_cfg": PULSE_SPEC,
    },
)

TWO_PULSE_RESET_SPEC = CfgSectionSpec(
    label="Two-Pulse Reset",
    fields={
        "type": ScalarSpec(label="Type", type=str, hidden=True),
        "pulse1_cfg": PULSE_SPEC,
        "pulse2_cfg": PULSE_SPEC,
    },
)

BATH_RESET_SPEC = CfgSectionSpec(
    label="Bath Reset",
    fields={
        "type": ScalarSpec(label="Type", type=str, hidden=True),
        "cavity_tone_cfg": PULSE_SPEC,
        "qubit_tone_cfg": PULSE_SPEC,
        "pi2_cfg": PULSE_SPEC,
    },
)
