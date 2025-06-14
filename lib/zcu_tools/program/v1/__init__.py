import warnings

warnings.warn("v1 is deprecated, use v2 instead")

from .base import MyAveragerProgram, MyNDAveragerProgram, MyRAveragerProgram
from .onetone import OneToneProgram, RGainOneToneProgram
from .singleshot import SingleShotProgram
from .time_exp import T1Program, T2EchoProgram, T2RamseyProgram
from .twotone import (
    PowerDepProgram,
    RFreqTwoToneProgram,
    RGainTwoToneProgram,
    TwoToneProgram,
)
