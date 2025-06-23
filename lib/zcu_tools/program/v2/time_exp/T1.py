import warnings
from typing import Any, Dict, List

from ..modular import ModularProgramV2
from ..modules import Module, Pulse, make_readout, make_reset


class T1Program(ModularProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        t1_spans = cfg["pi_pulse"].get("post_delay")
        if t1_spans is None:
            warnings.warn(
                "t1_spans is not set, are you sure you want to run T1 experiment?"
            )

        return [
            make_reset("reset", cfg=cfg["reset"]),
            Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
            make_readout("readout", cfg=cfg["readout"]),
        ]
