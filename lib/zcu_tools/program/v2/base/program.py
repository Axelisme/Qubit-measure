from typing import Any, Dict

from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2
from zcu_tools.program.base import MyProgram


class MyProgramV2(MyProgram, AveragerProgramV2):
    def __init__(self, soccfg: QickConfig, cfg: Dict[str, Any], **kwargs) -> None:
        # v2 program need to pass reps and final_delay to init
        super().__init__(
            soccfg,
            cfg=cfg,
            reps=cfg["reps"],
            initial_delay=0.0,
            final_delay=cfg["relax_delay"],
            **kwargs,
        )

    def _initialize_sweep(self, cfg: Dict[str, Any]) -> None:
        # add v2 sweep loops
        if "sweep" in cfg:
            for name, sweep in cfg["sweep"].items():
                self.add_loop(name, count=sweep["expts"])

    def _initialize(self, cfg: Dict[str, Any]) -> None:
        self._initialize_sweep(cfg)

    def acquire(self, soc, **kwargs) -> list:
        # v2 program need to pass rounds to acquire
        return super().acquire(soc, rounds=self.cfg["rounds"], **kwargs)

    def acquire_decimated(self, soc, **kwargs) -> list:
        # v2 program need to pass rounds to acquire_decimated
        return super().acquire_decimated(soc, rounds=self.cfg["rounds"], **kwargs)
