from __future__ import annotations

import qick.asm_v2 as qick_asm_v2

from .delay import Delay, SoftDelay, DelayAuto, Join
from .base import Module, ModuleCfg
from .loop import Repeat, SoftRepeat
from .pulse import Pulse, PulseCfg
from .readout import (
    DirectReadout,
    DirectReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
    ReadoutCfg,
)
from .reset import (
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
from .waveform import WaveformCfg

# TODO: waiting qick official implementation
# Monkey patching: implement __str__ and __repr__ methods for qick.asm_v2.QickParam


def param_repr(self) -> str:
    return f"QickParam({param2str(self)})"


def param_str(self) -> str:
    return param2str(self)


qick_asm_v2.QickParam.__repr__ = param_repr
qick_asm_v2.QickParam.__str__ = param_str
