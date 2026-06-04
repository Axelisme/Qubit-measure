"""Spec helper factories for common ModuleRefSpec combinations."""

from __future__ import annotations

from typing import Optional

from zcu_tools.gui.app.main.adapter import CfgSchema, ModuleRefSpec
from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec
from zcu_tools.gui.app.main.specs.readout import (
    make_direct_readout_spec,
    make_pulse_readout_spec,
)
from zcu_tools.gui.app.main.specs.reset import (
    make_bath_reset_spec,
    make_none_reset_spec,
    make_pulse_reset_spec,
    make_two_pulse_reset_spec,
)


def make_readout_module_spec(
    label: str = "Readout", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[make_direct_readout_spec(), make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


def make_pulse_readout_module_spec(
    label: str = "Readout", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


def make_pulse_module_spec(
    label: str = "Init Pulse", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[make_pulse_spec()],
        label=label,
        optional=optional,
    )


def make_reset_module_spec(
    label: str = "Reset", optional: bool = False
) -> ModuleRefSpec:
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


def schema_from_module(proposed: object | None) -> Optional[CfgSchema]:
    """Edit CfgSchema from a proposed module/waveform cfg (module-agnostic).

    Delegates to ``module_cfg_to_value``, which auto-routes waveform cfgs to
    ``waveform_cfg_to_value``. Returns None when ``proposed`` is None (no edit
    button shown). Unknown module types fast-fail (``module_cfg_to_value``
    raises) — deliberate.
    """
    if proposed is None:
        return None
    spec, value = module_cfg_to_value(proposed)
    return CfgSchema(spec=spec, value=value)
