"""Main-app policy facade for program module/waveform materialization."""

from __future__ import annotations

from typing import Any

from zcu_tools.gui.app.main.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.cfg import CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.measure_cfg import (
    PROGRAM_SHAPES,
    ProgramMaterializationPolicy,
    materialize_program_module,
    materialize_program_waveform,
)
from zcu_tools.program.v2.modules.base import AbsModuleCfg
from zcu_tools.program.v2.modules.waveform import AbsWaveformCfg

MAIN_PROGRAM_MATERIALIZATION_POLICY = ProgramMaterializationPolicy(
    spec_policy=MAIN_PROGRAM_SPEC_POLICY,
    allowed_module_discriminators=frozenset(
        shape.discriminator for shape in PROGRAM_SHAPES.modules()
    ),
    allowed_waveform_styles=frozenset(
        shape.discriminator for shape in PROGRAM_SHAPES.waveforms()
    ),
)

# The current Controller consumer still reads this private factory table. Phase 7c
# moves that consumer to a directly carried canonical shape; 7b does not mix in
# the RoleEntry/Controller migration.
_MODULE_SPEC_FACTORIES = {
    shape.discriminator: (lambda shape=shape: shape.make_spec(MAIN_PROGRAM_SPEC_POLICY))
    for shape in PROGRAM_SHAPES.modules()
}


def waveform_cfg_to_value(
    cfg_input: Any,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Convert a raw or typed waveform using the main-app spec policy."""

    if isinstance(cfg_input, AbsWaveformCfg):
        cfg = cfg_input.to_dict()
    elif isinstance(cfg_input, dict):
        cfg = cfg_input
    else:
        raise TypeError(f"Expected dict or AbsWaveformCfg, got {type(cfg_input)}")
    return materialize_program_waveform(cfg, MAIN_PROGRAM_MATERIALIZATION_POLICY)


def module_cfg_to_value(
    cfg_input: Any,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Convert any of the seven modules or six waveform shapes supported by main."""

    if isinstance(cfg_input, (AbsModuleCfg, AbsWaveformCfg)):
        cfg = cfg_input.to_dict()
    elif isinstance(cfg_input, dict):
        cfg = cfg_input
    else:
        raise TypeError(f"Expected dict or ModuleCfg, got {type(cfg_input)}")
    if "style" in cfg:
        return materialize_program_waveform(cfg, MAIN_PROGRAM_MATERIALIZATION_POLICY)
    return materialize_program_module(cfg, MAIN_PROGRAM_MATERIALIZATION_POLICY)
