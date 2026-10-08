"""Main-app policy facade for program module/waveform materialization."""

from __future__ import annotations

from typing import Any

from zcu_tools.cfg_model import AbsModuleCfg as LibraryModuleCfg
from zcu_tools.cfg_model import AbsWaveformCfg as LibraryWaveformCfg
from zcu_tools.experiment.cfg_editing import (
    PROGRAM_SHAPES,
    ProgramMaterializationPolicy,
    materialize_program_module,
    materialize_program_waveform,
)
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.cfg import CfgSectionSpec, CfgSectionValue
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


def waveform_cfg_to_value(
    cfg_input: Any,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Convert raw, library or runtime waveform data using the main-app policy.

    Typed inputs serialize through their owning to_dict contract once. The
    materializer owns field/default/reference representation and its failures.
    """

    if isinstance(cfg_input, (AbsWaveformCfg, LibraryWaveformCfg)):
        cfg = cfg_input.to_dict()
    elif isinstance(cfg_input, dict):
        cfg = cfg_input
    else:
        raise TypeError(f"Expected dict or AbsWaveformCfg, got {type(cfg_input)}")
    return materialize_program_waveform(cfg, MAIN_PROGRAM_MATERIALIZATION_POLICY)


def module_cfg_to_value(
    cfg_input: Any,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Convert raw, library or runtime cfg for main's seven modules/six waveforms.

    Typed inputs use their owner's to_dict once. Unsupported input, shape and
    malformed materialization errors propagate; no resource or hardware I/O.
    """

    if isinstance(
        cfg_input, (AbsModuleCfg, AbsWaveformCfg, LibraryModuleCfg, LibraryWaveformCfg)
    ):
        cfg = cfg_input.to_dict()
    elif isinstance(cfg_input, dict):
        cfg = cfg_input
    else:
        raise TypeError(f"Expected dict or ModuleCfg, got {type(cfg_input)}")
    if "style" in cfg:
        return materialize_program_waveform(cfg, MAIN_PROGRAM_MATERIALIZATION_POLICY)
    return materialize_program_module(cfg, MAIN_PROGRAM_MATERIALIZATION_POLICY)
