from typing import Any, Dict, List

from .modular import ModularProgramV2
from .modules import Module, make_readout, make_reset


class OneToneProgram(ModularProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return [
            make_reset("reset", cfg=cfg["reset"]),
            make_readout("readout", cfg=cfg["readout"]),
        ]
