"""Fresh CfgSectionSpec factories for reset module types."""

from __future__ import annotations

from zcu_tools.gui.adapter import CfgSectionSpec, LiteralSpec
from zcu_tools.gui.specs.pulse import make_pulse_spec


def make_none_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="None Reset",
        fields={
            "type": LiteralSpec("reset/none"),
        },
    )


def make_pulse_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Pulse Reset",
        fields={
            "type": LiteralSpec("reset/pulse"),
            "pulse_cfg": make_pulse_spec(),
        },
    )


def make_two_pulse_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Two-Pulse Reset",
        fields={
            "type": LiteralSpec("reset/two_pulse"),
            "pulse1_cfg": make_pulse_spec(),
            "pulse2_cfg": make_pulse_spec(),
        },
    )


def make_bath_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Bath Reset",
        fields={
            "type": LiteralSpec("reset/bath"),
            "cavity_tone_cfg": make_pulse_spec(),
            "qubit_tone_cfg": make_pulse_spec(),
            "pi2_cfg": make_pulse_spec(),
        },
    )
