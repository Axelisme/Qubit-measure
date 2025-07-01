import warnings
from typing import Any, Dict, List

from ..modular import ModularProgramV2
from ..modules import Module, Pulse, make_readout, make_reset


class T2RamseyProgram(ModularProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        t2r_spans = cfg["pi2_pulse1"].get("post_delay")
        if t2r_spans is None:
            warnings.warn(
                "t2r_spans is not set, are you sure you want to run T2ramsey experiment?"
            )

        return [
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse(name="pi2_pulse1", cfg=cfg["pi2_pulse1"]),
            Pulse(name="pi2_pulse2", cfg=cfg["pi2_pulse2"]),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ]
