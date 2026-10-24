from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Optional, TypeAlias, Union

import qick.asm_v2 as qick_asm_v2
from pydantic import BeforeValidator, Field, TypeAdapter

from zcu_tools import cfg_model as core_cfgs
from zcu_tools.cfg_model import CfgValidationContext
from zcu_tools.cfg_model import ModuleCfg as CoreModuleCfg

if TYPE_CHECKING:
    from zcu_tools.resources.context import LibrarySnapshot

from .base import AbsModuleCfg, Module, resolve_module_ref
from .computed_pulse import ComputedPulse
from .control import Branch, Repeat
from .delay import Delay, DelayAuto, Join, SoftDelay
from .dmem import LoadValue, LoadWord, ScanWith
from .pulse import Pulse, PulseCfg, TableLengthPulse
from .readout import (
    AbsReadout,
    AbsReadoutCfg,
    MuxReadout,
    MuxReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
    ReadoutCfg,
    TablePulseReadout,
)
from .reset import (
    AbsReset,
    AbsResetCfg,
    BathReset,
    BathResetCfg,
    PulseReset,
    PulseResetCfg,
    Reset,
    ResetCfg,
    TwoPulseReset,
    TwoPulseResetCfg,
)
from .util import param2str, round_timestamp
from .waveform import (
    AbsWaveform,
    AbsWaveformCfg,
    ArbWaveform,
    ArbWaveformCfg,
    ConstWaveform,
    ConstWaveformCfg,
    CosineWaveform,
    CosineWaveformCfg,
    DragWaveform,
    DragWaveformCfg,
    FlatTopWaveform,
    FlatTopWaveformCfg,
    GaussWaveform,
    GaussWaveformCfg,
    WaveformCfg,
    WaveformCfgFactory,
)

# QickParam lacks readable formatting in supported QICK versions; patch repr/str
# so diagnostics and warnings show sweep expressions.


ModuleCfg: TypeAlias = Annotated[
    ResetCfg | ReadoutCfg | PulseCfg,
    BeforeValidator(resolve_module_ref),
    Field(discriminator="type"),
]


class ModuleCfgFactory:
    @staticmethod
    def from_core(
        core_cfg: CoreModuleCfg,
        *,
        context: CfgValidationContext | None,
    ) -> ModuleCfg:
        """Convert a materialized core module tree to native runtime cfg.

        Resolve every nested arb through the supplied capture-bound
        context.assets.path_for, including pulses/readouts/resets and flat-top
        raise envelopes. Missing context/assets raises ValueError at arb use;
        key/missing-file/I/O failures propagate from the asset owner. No input
        is mutated and no provider/library/context is retained in native cfg.
        Non-arb trees permit context=None. Formula semantics are not interpreted.
        """
        if isinstance(core_cfg, core_cfgs.PulseCfg):
            return _pulse_from_core(core_cfg, context=context)
        if isinstance(core_cfg, core_cfgs.PulseReadoutCfg):
            fields = core_cfg.model_dump(mode="python", exclude={"waveform"})
            return PulseReadoutCfg.model_validate(
                {
                    **fields,
                    "waveform": WaveformCfgFactory.from_core(
                        core_cfg.waveform, context=context
                    ),
                }
            )
        if isinstance(core_cfg, core_cfgs.PulseResetCfg):
            return PulseResetCfg(
                pulse_cfg=_pulse_from_core(core_cfg.pulse_cfg, context=context)
            )
        if isinstance(core_cfg, core_cfgs.TwoPulseResetCfg):
            return TwoPulseResetCfg(
                pulse1_cfg=_pulse_from_core(core_cfg.pulse1_cfg, context=context),
                pulse2_cfg=_pulse_from_core(core_cfg.pulse2_cfg, context=context),
            )
        if isinstance(core_cfg, core_cfgs.BathResetCfg):
            return BathResetCfg(
                cavity_tone_cfg=_pulse_from_core(
                    core_cfg.cavity_tone_cfg, context=context
                ),
                qubit_tone_cfg=_pulse_from_core(
                    core_cfg.qubit_tone_cfg, context=context
                ),
                pi2_cfg=_pulse_from_core(core_cfg.pi2_cfg, context=context),
            )
        return TypeAdapter(ModuleCfg).validate_python(core_cfg.to_dict())

    @classmethod
    def from_raw(
        cls,
        raw: str | dict[str, Any] | AbsModuleCfg,
        *,
        ml: LibrarySnapshot | None = None,
    ) -> ModuleCfg:
        if isinstance(raw, str):
            if ml is None:
                raise ValueError(
                    f"Got module name '{raw}' but ModuleLibrary is not provided"
                )
            resolved_raw = ml.get_module(raw)
        else:
            resolved_raw = raw
        return TypeAdapter(ModuleCfg).validate_python(resolved_raw, context=dict(ml=ml))


def _pulse_from_core(
    core_cfg: core_cfgs.PulseCfg,
    *,
    context: CfgValidationContext | None,
) -> PulseCfg:
    """Translate pulse scalars and delegate envelope conversion to its owner."""
    fields = core_cfg.model_dump(mode="python", exclude={"waveform"}, exclude_none=True)
    return PulseCfg.model_validate(
        {
            **fields,
            "waveform": WaveformCfgFactory.from_core(
                core_cfg.waveform, context=context
            ),
        }
    )


def param_repr(self) -> str:
    return f"QickParam({param2str(self)})"


def param_str(self) -> str:
    return param2str(self)


qick_asm_v2.QickParam.__repr__ = param_repr
qick_asm_v2.QickParam.__str__ = param_str
