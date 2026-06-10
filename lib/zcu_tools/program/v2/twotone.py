from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel

from .modular import BaseCustomProgramV2, ProgramV2Cfg
from .modules import Module, Pulse, PulseCfg, Readout, ReadoutCfg, Reset, ResetCfg


class TwoToneModuleCfg(BaseModel):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class TwoToneCfg(ProgramV2Cfg):
    modules: TwoToneModuleCfg


class TwoToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: TwoToneCfg) -> Sequence[Module]:  # type: ignore
        modules = cfg.modules
        return [
            Reset("reset", cfg=modules.reset),
            Pulse("init_pulse", cfg=modules.init_pulse, tag="init_pulse"),
            Pulse("qubit_pulse", cfg=modules.qub_pulse, tag="qub_pulse"),
            Readout("readout", cfg=modules.readout),
        ]
