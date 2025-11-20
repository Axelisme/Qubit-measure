from typing import List

from typing_extensions import NotRequired

from .modular import BaseCustomProgramV2, ModularProgramCfg
from .modules import Module, Pulse, PulseCfg, Readout, ReadoutCfg, Reset, ResetCfg


class TwoToneProgramCfg(ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class TwoToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: TwoToneProgramCfg) -> List[Module]:
        return [
            Reset("reset", cfg=cfg.get("reset", {"type": "none"})),
            Pulse("init_pulse", cfg=cfg.get("init_pulse")),
            Pulse("qubit_pulse", cfg=cfg["qub_pulse"]),
            Readout("readout", cfg=cfg["readout"]),
        ]
