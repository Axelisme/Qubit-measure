from typing import Any, Dict, List

from .modular import ModularProgramV2
from .modules import Module, Pulse, make_readout, make_reset


class ResetProbeProgram(ModularProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return [
            make_reset("reset", cfg=cfg["reset"]),
            Pulse(name="init_pulse", cfg=cfg["init_pulse"]),
            make_reset("tested_reset", cfg=cfg["tested_reset"]),
            make_readout("readout", cfg=cfg["readout"]),
        ]
