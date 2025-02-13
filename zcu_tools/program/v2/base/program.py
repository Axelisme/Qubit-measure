from typing import Any, Dict

from qick.asm_v2 import AveragerProgramV2
from zcu_tools.program.base import MyProgram

from .readout import make_readout
from .reset import make_reset


class MyProgramV2(MyProgram, AveragerProgramV2):
    def __init__(self, soccfg, cfg: Dict[str, Any]):
        # v2 program need to pass reps and final_delay to init
        super().__init__(
            soccfg, cfg=cfg, reps=cfg["reps"], final_delay=cfg["adc"]["relax_delay"]
        )

    def _initialize(self, cfg):
        if "sweep" in cfg:
            # add loops
            for name, sweep in cfg["sweep"].items():
                self.add_loop(name, count=sweep["expts"])

    def parse_modules(self, cfg: dict):
        # reset and readout modules
        self.resetM = make_reset(cfg["dac"]["reset"])
        self.readoutM = make_readout(cfg["dac"]["readout"])
