from __future__ import annotations

from pydantic import BaseModel
from typing_extensions import Optional, Sequence

from .modular import BaseCustomProgramV2, ProgramV2Cfg
from .modules import Module, Pulse, PulseCfg, Readout, ReadoutCfg, Reset, ResetCfg


class OneToneModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
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
