import warnings
from typing import Any, Dict, List

from .modular import ModularProgramV2
from .modules import Module, Pulse, make_readout, make_reset


class ACStarkProgram(ModularProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        internal_delay = cfg["stark_pulse1"].get("post_delay")
        if internal_delay is not None:
            warnings.warn(
                "delay between stark_pulse1 and stark_pulse2 is not None, "
                "this will make two pulses are not overlapped. "
                "Make sure you know what you are doing."
            )

        return [
            make_reset("reset", cfg=cfg["reset"]),
            Pulse(name="stark_pulse1", cfg=cfg["stark_pulse1"]),
            Pulse(name="stark_pulse2", cfg=cfg["stark_pulse2"]),
            make_readout("readout", cfg=cfg["readout"]),
        ]
