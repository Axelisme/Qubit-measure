from typing import Any, Dict, List

from .modular import BaseCustomProgramV2
from .modules import Module, make_readout, make_reset


class OneToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return [
            make_reset("reset", reset_cfg=cfg.get("reset")),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ]
