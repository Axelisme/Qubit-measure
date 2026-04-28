from __future__ import annotations

import qick.asm_v2 as qick_asm_v2
from pydantic import BeforeValidator, Field, TypeAdapter
from typing_extensions import TYPE_CHECKING, Annotated, Any, Optional, TypeAlias, Union

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary

from .base import AbsModuleCfg, Module, resolve_module_ref
from .computed_pulse import ComputedPulse
from .control import Branch, Repeat, SoftRepeat
from .delay import Delay, DelayAuto, Join, SoftDelay
from .dmem import LoadValue, ScanWith
from .pulse import Pulse, PulseCfg
from .readout import (
    AbsReadout,
    AbsReadoutCfg,
    DirectReadout,
    DirectReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
    ReadoutCfg,
)
from .reset import (
    AbsReset,
    AbsResetCfg,
    BathReset,
    BathResetCfg,
    NoneReset,
    NoneResetCfg,
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
    ArbWaveformCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
    WaveformCfg,
    WaveformCfgFactory,
)

# TODO: waiting qick official implementation
# Monkey patching: implement __str__ and __repr__ methods for qick.asm_v2.QickParam


ModuleCfg: TypeAlias = Annotated[
    Union[ResetCfg, ReadoutCfg, PulseCfg],
    BeforeValidator(resolve_module_ref),
    Field(discriminator="type"),
]


class ModuleCfgFactory:
    @classmethod
    def from_raw(cls, raw: Any, *, ml: Optional[ModuleLibrary] = None) -> ModuleCfg:
        if isinstance(raw, str):
            if ml is None:
                raise ValueError("ModuleLibrary context not found")
            raw = ml.get_module(raw)
        ctx = {"ml": ml} if ml is not None else None
        return TypeAdapter(ModuleCfg).validate_python(raw, context=ctx)


def param_repr(self) -> str:
    return f"QickParam({param2str(self)})"


def param_str(self) -> str:
    return param2str(self)


qick_asm_v2.QickParam.__repr__ = param_repr
qick_asm_v2.QickParam.__str__ = param_str
