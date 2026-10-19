"""Module and depth-sweep settings shared by reference and interleaved RB."""

from typing import Annotated

from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import PulseCfg, ReadoutCfg, ResetCfg, SweepCfg


class RBModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    I_pulse: Annotated[PulseCfg | None, FieldMeta(label="Identity Pulse")] = None
    X90_pulse: Annotated[PulseCfg, FieldMeta(label="X90 Pulse")]
    X180_pulse: Annotated[PulseCfg, FieldMeta(label="X180 Pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class RBSweepCfg(ConfigBase):
    depth: Annotated[
        SweepCfg | list[int], FieldMeta(label="Clifford depth", decimals=0)
    ]
