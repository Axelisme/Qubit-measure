from __future__ import annotations

import qick.asm_v2 as qick_asm_v2

from .base import Module, ModuleCfg
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
    ArbWaveformCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
    WaveformCfg,
)

# TODO: waiting qick official implementation
# Monkey patching: implement __str__ and __repr__ methods for qick.asm_v2.QickParam


def param_repr(self) -> str:
    return f"QickParam({param2str(self)})"


def param_str(self) -> str:
    return param2str(self)


qick_asm_v2.QickParam.__repr__ = param_repr
qick_asm_v2.QickParam.__str__ = param_str

__all__ = [
    # base
    "Module",
    "ModuleCfg",
    # control
    "Branch",
    "Repeat",
    "SoftRepeat",
    # delay
    "Delay",
    "DelayAuto",
    "Join",
    "SoftDelay",
    # dmem
    "LoadValue",
    "ScanWith",
    # pulse
    "Pulse",
    "PulseCfg",
    # readout
    "AbsReadout",
    "AbsReadoutCfg",
    "DirectReadout",
    "DirectReadoutCfg",
    "PulseReadout",
    "PulseReadoutCfg",
    "Readout",
    "ReadoutCfg",
    # reset
    "AbsReset",
    "AbsResetCfg",
    "BathReset",
    "BathResetCfg",
    "NoneReset",
    "NoneResetCfg",
    "PulseReset",
    "PulseResetCfg",
    "Reset",
    "ResetCfg",
    "TwoPulseReset",
    "TwoPulseResetCfg",
    # util
    "param2str",
    "round_timestamp",
    # waveform
    "AbsWaveform",
    "ArbWaveformCfg",
    "ConstWaveformCfg",
    "CosineWaveformCfg",
    "DragWaveformCfg",
    "FlatTopWaveformCfg",
    "GaussWaveformCfg",
    "WaveformCfg",
]
