from typing import Any, Dict, List

from .modular import ModularProgramV2
from .modules import Module, Pulse, make_readout, make_reset


class TwoToneProgram(ModularProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return [
            make_reset("reset", cfg=cfg["reset"]),
            Pulse(name="qubit_pulse", cfg=cfg["qub_pulse"]),
            make_readout("readout", cfg=cfg["readout"]),
        ]
