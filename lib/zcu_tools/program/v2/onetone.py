from typing import List

from typing_extensions import NotRequired

from .modular import BaseCustomProgramV2, ModularProgramCfg
from .modules import Module, Pulse, PulseCfg, Readout, ReadoutCfg, Reset, ResetCfg


class OneToneProgramCfg(ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    readout: ReadoutCfg


class OneToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: OneToneProgramCfg) -> List[Module]:
        return [
            Reset("reset", cfg=cfg.get("reset", {"type": "none"})),
            Pulse("init_pulse", cfg=cfg.get("init_pulse")),
            Readout("readout", cfg=cfg["readout"]),
        ]
