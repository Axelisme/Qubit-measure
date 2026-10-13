"""Shared value types for the experiment Run mapping boundary."""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import ModuleCfg, ModuleSpec, WaveformCfg, WaveformSpec
from zcu_tools.program.v2 import SweepCfg

type ExperimentScalar = bool | int | float | complex | str
type ExperimentCfgValue = (
    None
    | ExperimentScalar
    | QickParam
    | SweepCfg
    | ModuleSpec
    | WaveformSpec
    | ModuleCfg
    | WaveformCfg
    | NDArray[np.float64]
    | Sequence[ExperimentCfgValue]
    | Mapping[str, ExperimentCfgValue]
)
