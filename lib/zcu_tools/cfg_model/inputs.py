"""Recursive, unvalidated inputs accepted only at cfg factory boundaries."""

from collections.abc import Mapping, Sequence

from qick.asm_v2 import QickParam

from .base import ConfigBase

type CfgInputValue = (
    None
    | bool
    | int
    | float
    | complex
    | str
    | QickParam
    | ConfigBase
    | Sequence[CfgInputValue]
    | Mapping[str, CfgInputValue]
)
type CfgInputMap = Mapping[str, CfgInputValue]
