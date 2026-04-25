from __future__ import annotations

from zcu_tools.config import ConfigBase


class SweepCfg(ConfigBase):
    start: float
    stop: float
    expts: int
    step: float
