import warnings
from typing import Any, Dict, List

from ..modular import ModularProgramV2
from ..modules import Module, Pulse, make_readout, make_reset


class T2EchoProgram(ModularProgramV2):
    def make_modules(self, cfg: Dict[str, Any]) -> List[Module]:
        t2e_half1 = cfg["pi2_pulse1"].get("post_delay")
        t2e_half2 = cfg["pi2_pulse2"].get("post_delay")
        if t2e_half1 is None:
            warnings.warn(
                "t2e_half1 is not set, this will make two pulses are overlapped."
                "Make sure you know what you are doing."
            )
        if t2e_half2 is None:
            warnings.warn(
                "t2e_half2 is not set, this will make two pulses are overlapped."
                "Make sure you know what you are doing."
            )
        if t2e_half1 != t2e_half2:
            warnings.warn(
                "t2e_half1 and t2e_half2 are not the same, "
                "this is different from the original T2echo experiment."
                "Please make sure you know what you are doing."
            )

        return [
            make_reset("reset", cfg=cfg["reset"]),
            Pulse(name="pi2_pulse1", cfg=cfg["pi2_pulse1"]),
            Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
            Pulse(name="pi2_pulse2", cfg=cfg["pi2_pulse2"]),
            make_readout("readout", cfg=cfg["readout"]),
        ]
