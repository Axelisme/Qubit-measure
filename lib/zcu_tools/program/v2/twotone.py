from typing_extensions import NotRequired, Sequence, TypedDict

from .modular import BaseCustomProgramV2, ModularProgramCfg
from .modules import Module, Pulse, PulseCfg, Readout, ReadoutCfg, Reset, ResetCfg


class TwoToneModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class TwoToneCfg(ModularProgramCfg):
    modules: TwoToneModuleCfg


class TwoToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: TwoToneCfg) -> Sequence[Module]:
        modules = cfg["modules"]
        return [
            Reset("reset", cfg=modules.get("reset")),
            Pulse("init_pulse", cfg=modules.get("init_pulse")),
            Pulse("qubit_pulse", cfg=modules["qub_pulse"]),
            Readout("readout", cfg=modules["readout"]),
        ]
