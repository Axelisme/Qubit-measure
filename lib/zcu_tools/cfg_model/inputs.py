"""Recursive, unvalidated inputs accepted only at cfg factory boundaries."""

from qick.asm_v2 import QickParam

from .module import ModuleCfg
from .waveform import WaveformCfg

type CfgInputValue = (
    None
    | bool
    | int
    | float
    | complex
    | str
    | QickParam
    | ModuleCfg
    | WaveformCfg
    | list[CfgInputValue]
    | dict[str, CfgInputValue]
)
type CfgInputMap = dict[str, CfgInputValue]
