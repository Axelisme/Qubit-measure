from .base import Delay, Module, NonBlocking
from .loop import Repeat
from .pulse import Pulse, PulseCfg, check_block_mode
from .readout import (
    AbsReadout,
    BaseReadout,
    Readout,
    ReadoutCfg,
    TriggerCfg,
    TriggerReadout,
)
from .reset import (
    AbsReset,
    BathReset,
    NoneReset,
    PulseReset,
    Reset,
    ResetCfg,
    TwoPulseReset,
)
from .waveform import WaveformCfg
from .util import param2str

# TODO: monkey patching fix
# Implement __str__ and __repr__ methods for qick.asm_v2.QickParam

import qick.asm_v2 as qick_asm_v2


def param_repr(self) -> str:
    return f"QickParam({param2str(self)})"


def param_str(self) -> str:
    return param2str(self)


qick_asm_v2.QickParam.__repr__ = param_repr
qick_asm_v2.QickParam.__str__ = param_str
