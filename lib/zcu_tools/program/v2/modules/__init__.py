from __future__ import annotations

import qick.asm_v2 as qick_asm_v2

from .base import Delay, Module, ModuleCfg, NonBlocking
from .loop import Repeat
from .pulse import Pulse, PulseCfg, check_block_mode
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
from .util import param2str
from .waveform import WaveformCfg

# TODO: waiting qick official implementation
# Monkey patching: implement __str__ and __repr__ methods for qick.asm_v2.QickParam


def param_repr(self) -> str:
    return f"QickParam({param2str(self)})"


def param_str(self) -> str:
    return param2str(self)


qick_asm_v2.QickParam.__repr__ = param_repr
qick_asm_v2.QickParam.__str__ = param_str
