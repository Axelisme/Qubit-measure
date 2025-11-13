from typing import Any, Dict, List

from .modular import BaseCustomProgramV2
from .modules import Module, Pulse, Readout, Reset


class TwoToneProgram(BaseCustomProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        return [
            Reset("reset", cfg=cfg.get("reset", {"type": "none"})),
            Pulse("init_pulse", cfg=cfg.get("init_pulse")),
            Pulse("qubit_pulse", cfg=cfg["qub_pulse"]),
            Readout("readout", cfg=cfg["readout"]),
        ]
