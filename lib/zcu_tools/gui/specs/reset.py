"""Fresh CfgSectionSpec factories for reset module types."""

from __future__ import annotations

from zcu_tools.gui.adapter import CfgSectionSpec, ScalarSpec
from zcu_tools.gui.specs.pulse import make_pulse_spec


def make_none_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="None Reset",
        fields={
            "type": ScalarSpec(label="Type", type=str, hidden=True),
        },
    )


def make_pulse_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Pulse Reset",
        fields={
            "type": ScalarSpec(label="Type", type=str, hidden=True),
            "pulse_cfg": make_pulse_spec(),
        },
    )


def make_two_pulse_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Two-Pulse Reset",
        fields={
            "type": ScalarSpec(label="Type", type=str, hidden=True),
            "pulse1_cfg": make_pulse_spec(),
            "pulse2_cfg": make_pulse_spec(),
        },
    )


def make_bath_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Bath Reset",
        fields={
            "type": ScalarSpec(label="Type", type=str, hidden=True),
            "cavity_tone_cfg": make_pulse_spec(),
            "qubit_tone_cfg": make_pulse_spec(),
            "pi2_cfg": make_pulse_spec(),
        },
    )
