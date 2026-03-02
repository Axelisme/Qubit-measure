from typing_extensions import NotRequired, Sequence, TypedDict

from .modular import BaseCustomProgramV2, ModularProgramCfg
from .modules import Module, Pulse, PulseCfg, Readout, ReadoutCfg, Reset, ResetCfg


class OneToneModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    readout: ReadoutCfg


class OneToneCfg(ModularProgramCfg):
    modules: OneToneModuleCfg


class OneToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: OneToneCfg) -> Sequence[Module]:
        modules = cfg["modules"]
        return [
            Reset("reset", cfg=modules.get("reset", {"type": "none"})),
            Pulse("init_pulse", cfg=modules.get("init_pulse")),
            Readout("readout", cfg=modules["readout"]),
        ]
