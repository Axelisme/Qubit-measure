from typing import Any, Dict, override

from qick.asm_v2 import AveragerProgramV2
from zcu_tools.program.base import MyProgram

from .readout import make_readout
from .reset import make_reset


class MyProgramV2(MyProgram, AveragerProgramV2):
    def __init__(self, soccfg, cfg: Dict[str, Any], **kwargs):
        # v2 program need to pass reps and final_delay to init
        super().__init__(
            soccfg,
            cfg=cfg,
            reps=cfg["reps"],
            final_delay=cfg["adc"]["relax_delay"],
            **kwargs,
        )

    @override
    def _parse_cfg(self, cfg: Dict[str, Any]):
        self.resetM = make_reset(cfg["dac"]["reset"])
        self.readoutM = make_readout(cfg["dac"]["readout"])
        return super()._parse_cfg(cfg)

    @override
    def _initialize(self, cfg: Dict[str, Any]):
        # add sweep loops
        if "sweep" in cfg:
            for name, sweep in cfg["sweep"].items():
                self.add_loop(name, count=sweep["expts"])

        # initialize reset and readout modules
        self.resetM.init(self)
        self.readoutM.init(self)

    @override
    def acquire(self, soc, **kwargs):
        return super().acquire(soc, soft_avgs=self.cfg["soft_avgs"], **kwargs)

    @override
    def acquire_decimated(self, soc, **kwargs):
        return super().acquire_decimated(soc, soft_avgs=self.cfg["soft_avgs"], **kwargs)


class DoNothingProgramV2(MyProgramV2):
    def _body(self, _):
        # only acquire
        ro_ch = self.adc["chs"][0]

        self.send_readoutconfig(ro_ch, "readout_adc", t=0)
        self.delay_auto(t=self.adc["trig_offset"], ros=False, tag="trig_offset")
        self.trigger([ro_ch], t=None)
