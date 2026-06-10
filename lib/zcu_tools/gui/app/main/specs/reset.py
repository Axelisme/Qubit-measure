"""Fresh CfgSectionSpec factories for reset module types."""

from __future__ import annotations

from zcu_tools.gui.app.main.adapter import CfgSectionSpec, LiteralSpec

from .pulse import make_pulse_spec


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
            "pulse1_cfg": make_pulse_spec(label="Pulse 1"),
            "pulse2_cfg": make_pulse_spec(label="Pulse 2"),
        },
    )


def make_bath_reset_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Bath Reset",
        fields={
            "type": LiteralSpec("reset/bath"),
            "cavity_tone_cfg": make_pulse_spec(label="Cavity Tone"),
            "qubit_tone_cfg": make_pulse_spec(label="Qubit Tone"),
            "pi2_cfg": make_pulse_spec(label="Pi/2 Pulse"),
        },
    )
