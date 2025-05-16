from typing import Any, Dict

from myqick.asm_v2 import AveragerProgramV2
from zcu_tools.program.base import MyProgram

from .readout import make_readout
from .reset import make_reset
from .simulate import SimulateV2


class MyProgramV2(SimulateV2, MyProgram, AveragerProgramV2):
    """
    Convert general config to qick v2 specific api calls
    """

    def __init__(self, soccfg, cfg: Dict[str, Any], **kwargs):
        # v2 program need to pass reps and final_delay to init
        super().__init__(
            soccfg,
            cfg=cfg,
            reps=cfg["reps"],
            initial_delay=0.0,
            final_delay=cfg["adc"]["relax_delay"],
            **kwargs,
        )

    def _parse_cfg(self, cfg: Dict[str, Any]):
        # instaniate v2 reset and readout modules
        self.resetM = make_reset(cfg["dac"]["reset"])
        self.readoutM = make_readout(cfg["dac"]["readout"])
        return super()._parse_cfg(cfg)

    def _initialize(self, cfg: Dict[str, Any]):
        # add v2 sweep loops
        if "sweep" in cfg:
            for name, sweep in cfg["sweep"].items():
                self.add_loop(name, count=sweep["expts"])

        # initialize v2 reset and readout modules
        self.resetM.init(self)
        self.readoutM.init(self)

    def acquire(self, soc, **kwargs):
        return super().acquire(soc, soft_avgs=self.cfg["soft_avgs"], **kwargs)

    def acquire_decimated(self, soc, **kwargs):
        return super().acquire_decimated(soc, soft_avgs=self.cfg["soft_avgs"], **kwargs)
