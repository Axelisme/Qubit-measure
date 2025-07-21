from typing import Any, Dict, List

from .modular import BaseCustomProgramV2
from .modules import Module, Pulse, make_readout, make_reset


class TwoToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return [
            make_reset("reset", reset_cfg=cfg.get("reset")),
            Pulse(name="qubit_pulse", cfg=cfg["qub_pulse"]),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ]
