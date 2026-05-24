"""Spec helper factories for common ModuleRefSpec combinations."""

from __future__ import annotations

from zcu_tools.gui.adapter import ModuleRefSpec
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


def make_readout_ref_spec(
    label: str = "Readout", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[make_direct_readout_spec(), make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


def make_pulse_readout_ref_spec(
    label: str = "Readout", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


def make_pulse_ref_spec(
    label: str = "Init Pulse", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[make_pulse_spec()],
        label=label,
        optional=optional,
    )


def make_reset_ref_spec(label: str = "Reset", optional: bool = False) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[
            make_none_reset_spec(),
            make_pulse_reset_spec(),
            make_two_pulse_reset_spec(),
            make_bath_reset_spec(),
        ],
        label=label,
        optional=optional,
    )
