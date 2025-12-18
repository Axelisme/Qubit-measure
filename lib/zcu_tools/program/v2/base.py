from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2
from typing_extensions import TypedDict

from zcu_tools.program.base import MyProgram

from .modules.registry import PulseRegistry


class ProgramV2Cfg(TypedDict):
    reps: int
    rounds: int
    relax_delay: float


class MyProgramV2(MyProgram, AveragerProgramV2):
    def __init__(self, soccfg: QickConfig, cfg: ProgramV2Cfg, **kwargs) -> None:
        # v2 program need to pass reps and final_delay to init
        self.pulse_registry = PulseRegistry()
        super().__init__(
            soccfg,
            cfg=dict(cfg),
            reps=cfg["reps"],
            initial_delay=0.0,
            final_wait=cfg.get("final_wait", 0.0),
            final_delay=cfg["relax_delay"],
            **kwargs,
        )

    def acquire(self, soc, **kwargs):
        # v2 program need to pass rounds to acquire
        return super().acquire(soc, rounds=self.cfg["rounds"], **kwargs)

    def acquire_decimated(self, soc, **kwargs):
        # v2 program need to pass rounds to acquire_decimated
        return super().acquire_decimated(soc, rounds=self.cfg["rounds"], **kwargs)
