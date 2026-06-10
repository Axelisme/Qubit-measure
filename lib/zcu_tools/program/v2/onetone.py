from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel

from .modular import BaseCustomProgramV2, ProgramV2Cfg
from .modules import Module, Pulse, PulseCfg, Readout, ReadoutCfg, Reset, ResetCfg


class OneToneModuleCfg(BaseModel):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    readout: ReadoutCfg


class OneToneCfg(ProgramV2Cfg):
    modules: OneToneModuleCfg


class OneToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: OneToneCfg) -> Sequence[Module]:  # type: ignore
        modules = cfg.modules
        return [
            Reset("reset", cfg=modules.reset),
            Pulse("init_pulse", cfg=modules.init_pulse, tag="init_pulse"),
            Readout("readout", cfg=modules.readout),
        ]
